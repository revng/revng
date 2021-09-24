/// \file StackAnalysis.cpp
/// \brief Implementation of the stack analysis, which provides information
///        about function boundaries, basic block types, arguments and return
///        values.

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <climits>
#include <cstdio>
#include <fstream>
#include <map>
#include <sstream>
#include <vector>

#include "llvm/ADT/DepthFirstIterator.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/ADT/SCCIterator.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Analysis/BasicAliasAnalysis.h"
#include "llvm/Analysis/ScopedNoAliasAA.h"
#include "llvm/CodeGen/UnreachableBlockElim.h"
#include "llvm/IR/Attributes.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/InstCombine/InstCombine.h"
#include "llvm/Transforms/Scalar/EarlyCSE.h"
#include "llvm/Transforms/Scalar/GVN.h"
#include "llvm/Transforms/Scalar/JumpThreading.h"
#include "llvm/Transforms/Scalar/MergedLoadStoreMotion.h"
#include "llvm/Transforms/Scalar/SROA.h"
#include "llvm/Transforms/Scalar/SimplifyCFG.h"
#include "llvm/Transforms/Utils.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/Transforms/Utils/CodeExtractor.h"
#include "llvm/Transforms/Utils/Mem2Reg.h"

#include "revng/ADT/KeyedObjectTraits.h"
#include "revng/ADT/Queue.h"
#include "revng/ADT/SortedVector.h"
#include "revng/ADT/ZipMapIterator.h"
#include "revng/BasicAnalyses/RemoveHelperCalls.h"
#include "revng/BasicAnalyses/RemoveNewPCCalls.h"
#include "revng/Model/Binary.h"
#include "revng/StackAnalysis/StackAnalysis.h"
#include "revng/Support/Assert.h"
#include "revng/Support/CommandLine.h"
#include "revng/Support/IRHelpers.h"
#include "revng/Support/MetaAddress.h"

#include "ABIAnalyses/ABIAnalysis.h"
#include "Cache.h"
#include "InterproceduralAnalysis.h"
#include "Intraprocedural.h"

using llvm::ArrayRef;
using llvm::BasicBlock;
using llvm::Function;
using llvm::GlobalVariable;
using llvm::Instruction;
using llvm::Module;
using llvm::raw_fd_ostream;
using llvm::RegisterPass;
using llvm::SmallVectorImpl;
using llvm::Type;

using FunctionEdgeTypeValue = model::FunctionEdgeType::Values;
using FunctionTypeValue = model::FunctionType::Values;
using GCBI = GeneratedCodeBasicInfo;

using namespace llvm::cl;

static Logger<> CFEPLog("cfep");
static Logger<> ClobberedLog("clobbered");
static Logger<> StackAnalysisLog("stackanalysis");

struct BasicBlockNodeData {
  BasicBlockNodeData(llvm::BasicBlock *BB) : BB(BB){};
  llvm::BasicBlock *BB;
};
using BasicBlockNode = BidirectionalNode<BasicBlockNodeData>;
using SmallCallGraph = GenericGraph<BasicBlockNode>;

template<>
struct llvm::DOTGraphTraits<SmallCallGraph *>
  : public llvm::DefaultDOTGraphTraits {
  using EdgeIterator = llvm::GraphTraits<SmallCallGraph *>::ChildIteratorType;
  DOTGraphTraits(bool IsSimple = false) : DefaultDOTGraphTraits(IsSimple) {}

  static std::string
  getNodeLabel(const BasicBlockNode *Node, const SmallCallGraph *Graph) {
    if (Node->BB == nullptr)
      return "null";
    return Node->BB->getName().str();
  }

  static std::string getEdgeAttributes(const BasicBlockNode *Node,
                                       const EdgeIterator EI,
                                       const SmallCallGraph *Graph) {
    return "color=black,style=dashed";
  }
};

namespace StackAnalysis {

const std::set<llvm::GlobalVariable *> EmptyCSVSet;

char StackAnalysis::ID = 0;

using RegisterABI = RegisterPass<StackAnalysis>;
static RegisterABI Y("abi-analysis", "ABI Analysis Pass", true, true);

static opt<std::string> ABIAnalysisOutputPath("abi-analysis-output",
                                              desc("Destination path for the "
                                                   "ABI Analysis Pass"),
                                              value_desc("path"),
                                              cat(MainCategory));

static opt<std::string> CallGraphOutputPath("cg-output",
                                            desc("Dump to disk the recovered "
                                                 "call graph."),
                                            value_desc("filename"));

static opt<std::string> IndirectBranchInfoSummaryPath("indirect-branch-info-"
                                                      "summary",
                                                      desc("Write the results "
                                                           "of SA2 on disk."),
                                                      value_desc("filename"));

static opt<std::string> AAWriterPath("aa-writer",
                                     desc("Dump to disk the outlined functions "
                                          "with annotated alias info."),
                                     value_desc("filename"));

/// Candidate Function Entry Points structure.
struct CFEP {
  CFEP(llvm::BasicBlock *Entry, bool Force) : Entry(Entry), Force(Force) {}

  llvm::BasicBlock *Entry;
  bool Force;
};

/// A summary of the analysis of a function.
///
/// For each function detected, the following information are included:
/// its type ("regular", "noreturn" or "fake"), which ABI registers are
/// overwritten, its control-flow graph, and an elected stack offset (to
/// tell if the stack pointer is restored at its original position).
struct FunctionSummary {
public:
  model::FunctionType::Values Type;
  std::set<llvm::GlobalVariable *> ClobberedRegisters;
  ABIAnalyses::ABIAnalysesResults ABIResults;
  SortedVector<model::BasicBlock> CFG;
  std::optional<int64_t> ElectedFSO;
  llvm::Function *FakeFunction;

public:
  FunctionSummary(model::FunctionType::Values Type,
                  std::set<llvm::GlobalVariable *> ClobberedRegisters,
                  ABIAnalyses::ABIAnalysesResults ABIResults,
                  SortedVector<model::BasicBlock> CFG,
                  std::optional<int64_t> ElectedFSO,
                  llvm::Function *FakeFunction) :
    Type(Type),
    ClobberedRegisters(std::move(ClobberedRegisters)),
    ABIResults(std::move(ABIResults)),
    CFG(std::move(CFG)),
    ElectedFSO(ElectedFSO),
    FakeFunction(FakeFunction) {
    if (FakeFunction != nullptr)
      revng_assert(CFG.empty());
  }

  FunctionSummary() = delete;
  FunctionSummary(const FunctionSummary &) = delete;
  FunctionSummary(FunctionSummary &&) = default;
  FunctionSummary &operator=(const FunctionSummary &) = delete;
  FunctionSummary &operator=(FunctionSummary &&) = default;

public:
  static bool compare(const FunctionSummary &Old, const FunctionSummary &New) {
    if (New.Type == Old.Type)
      return std::includes(Old.ClobberedRegisters.begin(),
                           Old.ClobberedRegisters.end(),
                           New.ClobberedRegisters.begin(),
                           New.ClobberedRegisters.end());

    return New.Type <= Old.Type;
  }

  void dump() const debug_function { dump(dbg); }

  template<typename T>
  void dump(T &Output) const {
    Output << "Dumping summary \n"
           << "  Type: " << Type << "\n"
           << "  ElectedFSO: " << (ElectedFSO.has_value() ? *ElectedFSO : -1)
           << "\n"
           << "  Clobbered registers: \n";

    for (auto *Reg : ClobberedRegisters)
      Output << "    " << Reg->getName().str() << "\n";

    Output << "  ABI info: \n";
    ABIResults.dump();
  }
};

/// A cache holding the results of the analyses of functions.
///
/// Leaf subroutines are analyzed first; non-leaf afterwards. The whole
/// process is repeated until a fixed point is reached, and no further
/// refinement can be achieved. The cache can be queried through specific
/// methods to retrieve the stored information, acting as a oracle.
class FunctionAnalysisResults {
private:
  /// For each function, the result of the intraprocedural analysis
  std::map<MetaAddress, FunctionSummary> FunctionsBucket;
  FunctionSummary DefaultSummary;

public:
  FunctionAnalysisResults(FunctionSummary DefaultSummary) :
    DefaultSummary(std::move(DefaultSummary)) {}

  FunctionSummary &at(MetaAddress PC) { return FunctionsBucket.at(PC); }

  const FunctionSummary &at(MetaAddress PC) const {
    return FunctionsBucket.at(PC);
  }

  model::FunctionType::Values getFunctionType(MetaAddress PC) const {
    return get(PC).Type;
  }

  bool isFakeFunction(MetaAddress PC) const {
    return getFunctionType(PC) == model::FunctionType::Values::Fake;
  }

  llvm::Function *getFakeFunction(MetaAddress PC) const {
    return get(PC).FakeFunction;
  }

  const auto &getRegistersClobbered(MetaAddress PC) const {
    return get(PC).ClobberedRegisters;
  }

  std::optional<uint64_t> getElectedFSO(MetaAddress PC) const {
    return get(PC).ElectedFSO;
  }

  bool registerFunction(MetaAddress PC, FunctionSummary &&F) {
    revng_assert(PC.isValid());
    auto It = FunctionsBucket.find(PC);
    if (It != FunctionsBucket.end()) {
      bool Changed = FunctionSummary::compare(It->second, F);
      It->second = std::move(F);
      return not Changed;
    } else {
      FunctionsBucket.emplace(PC, std::move(F));
      return true;
    }
  }

private:
  const FunctionSummary &get(MetaAddress PC) const {
    auto It = FunctionsBucket.find(PC);
    if (It != FunctionsBucket.end())
      return It->second;
    return DefaultSummary;
  }
};

/// An outlined function helper object.
struct OutlinedFunction {
  /// The actual LLVM outlined function
  llvm::Function *F = nullptr;
  /// The marker that detects returns and regular jumps
  llvm::Function *IndirectBranchInfoMarker = nullptr;
  llvm::BasicBlock *AnyPCCloned = nullptr;
  llvm::BasicBlock *UnexpectedPCCloned = nullptr;

  OutlinedFunction() = default;
  OutlinedFunction(const OutlinedFunction &Other) = delete;
  OutlinedFunction(OutlinedFunction &&Other) : F(Other.F) { Other.F = nullptr; }
  OutlinedFunction &operator=(OutlinedFunction &&) = delete;
  OutlinedFunction &operator=(const OutlinedFunction &) = delete;

  llvm::Function *extractFunction() {
    auto *ToReturn = F;
    F = nullptr;
    return ToReturn;
  }

  ~OutlinedFunction() {
    if (F != nullptr) {
      revng_assert(F->use_empty()
                   && "Failed to remove all users of the outlined function.");
      F->eraseFromParent();
    }

    if (IndirectBranchInfoMarker != nullptr)
      IndirectBranchInfoMarker->eraseFromParent();
  }
};

struct TemporaryOpaqueFunction {
  llvm::Function *F = nullptr;
  llvm::FunctionType *FTy;
  llvm::StringRef Name;
  llvm::Module *M;

  TemporaryOpaqueFunction(llvm::FunctionType *FTy,
                          llvm::StringRef Name,
                          llvm::Module *M) :
    FTy(FTy), Name(Name), M(M) {
    F = Function::Create(FTy, llvm::GlobalValue::ExternalLinkage, Name, M);

    revng_assert(F != nullptr);
    F->addFnAttr(llvm::Attribute::ReadOnly);
    F->addFnAttr(llvm::Attribute::NoUnwind);
    F->addFnAttr(llvm::Attribute::WillReturn);
  }

  ~TemporaryOpaqueFunction() {
    if (F != nullptr) {
      revng_assert(F->use_empty()
                   && "Failed to remove all users of the temporary opaque "
                      "function.");
      F->eraseFromParent();
    }
  }
};

/// An intraprocedural analysis storage.
///
/// Implementation of the intraprocedural stack analysis. It holds the
/// necessary information to detect the boundaries of a function, track
/// how the stack evolves within those functions, and detect the
/// callee-saved registers.
template<class FunctionOracle>
class CFEPAnalyzer {
private:
  llvm::Module &M;
  llvm::LLVMContext &Context;
  GeneratedCodeBasicInfo *GCBI;
  FunctionOracle &Oracle;
  ArrayRef<GlobalVariable *> ABIRegisters;
  /// PreHookMarker and PostHookMarker mark the presence of an original
  /// function call, and surround a basic block containing the registers
  /// clobbered by the function called. They take the MetaAddress of the
  /// callee and the call-site.
  TemporaryOpaqueFunction PreHookMarker;
  TemporaryOpaqueFunction PostHookMarker;
  TemporaryOpaqueFunction RetHookMarker;
  /// UnexpectedPCMarker is used to indicate that `unexpectedpc` basic
  /// block of fake functions need to be adjusted to jump to
  /// `unexpectedpc` of their caller.
  TemporaryOpaqueFunction UnexpectedPCMarker;
  std::unique_ptr<raw_fd_ostream> OutputIBI;
  std::unique_ptr<raw_fd_ostream> OutputAAWriter;
  OpaqueFunctionsPool<llvm::StringRef> RegistersClobberedPool;
  OpaqueFunctionsPool<llvm::Type *> OpaqueBranchConditionsPool;
  const llvm::CodeExtractorAnalysisCache CEAC;
  const ProgramCounterHandler *PCH;

public:
  CFEPAnalyzer(llvm::Module &,
               GeneratedCodeBasicInfo *GCBI,
               FunctionOracle &,
               ArrayRef<GlobalVariable *>);

public:
  /// The `analyze` method is the entry point of the intraprocedural analysis,
  /// and it is called on each CFEP until a fixed point is reached. It is
  /// responsible for performing the whole computation.
  FunctionSummary analyze(llvm::BasicBlock *BB);

private:
  OutlinedFunction outlineFunction(llvm::BasicBlock *BB);
  void integrateFunctionCallee(llvm::BasicBlock *BB, MetaAddress);
  SortedVector<model::BasicBlock> collectDirectCFG(OutlinedFunction *F);
  void initMarkersForABI(OutlinedFunction *F,
                         llvm::SmallVectorImpl<Instruction *> &,
                         llvm::IRBuilder<> &);
  std::set<llvm::GlobalVariable *> findWrittenRegisters(llvm::Function *F);
  void createIBIMarker(OutlinedFunction *F,
                       llvm::SmallVectorImpl<Instruction *> &,
                       llvm::IRBuilder<> &);
  void opaqueBranchConditions(llvm::Function *F, llvm::IRBuilder<> &);
  void materializePCValues(llvm::Function *F, llvm::IRBuilder<> &);
  void runOptimizationPipeline(llvm::Function *F);
  FunctionSummary milkInfo(OutlinedFunction *F,
                           SortedVector<model::BasicBlock> &,
                           ABIAnalyses::ABIAnalysesResults &,
                           const std::set<llvm::GlobalVariable *> &);
  llvm::Function *createFakeFunction(llvm::BasicBlock *BB);

private:
  static auto *markerType(llvm::Module &M) {
    return llvm::FunctionType::get(Type::getVoidTy(M.getContext()),
                                   { MetaAddress::getStruct(&M),
                                     MetaAddress::getStruct(&M) },
                                   false);
  }

  static auto *unexpectedPCMarkerType(llvm::Module &M) {
    return llvm::FunctionType::get(Type::getVoidTy(M.getContext()), false);
  }
};

using TOF = TemporaryOpaqueFunction;

template<class FunctionOracle>
CFEPAnalyzer<FunctionOracle>::CFEPAnalyzer(llvm::Module &M,
                                           GeneratedCodeBasicInfo *GCBI,
                                           FunctionOracle &Oracle,
                                           ArrayRef<GlobalVariable *> ABIRegs) :
  M(M),
  Context(M.getContext()),
  GCBI(GCBI),
  Oracle(Oracle),
  ABIRegisters(ABIRegs),
  // Initialize hook markers for subsequent ABI analyses on function calls
  PreHookMarker(TOF(markerType(M), "precall_hook", &M)),
  PostHookMarker(TOF(markerType(M), "postcall_hook", &M)),
  RetHookMarker(TOF(markerType(M), "retcall_hook", &M)),
  // Initialize marker to adjust `unexpectedpc` basic block for fake functions
  UnexpectedPCMarker(TOF(unexpectedPCMarkerType(M), "unexpectedpc_hook", &M)),
  RegistersClobberedPool(&M, false),
  OpaqueBranchConditionsPool(&M, false),
  // Initialize the cache for the `CodeExtractor` analysis on `root`
  CEAC(llvm::CodeExtractorAnalysisCache(*M.getFunction("root"))),
  PCH(GCBI->programCounterHandler()) {

  // Open streams for dumping results
  if (IndirectBranchInfoSummaryPath.getNumOccurrences() == 1) {
    std::ifstream File(IndirectBranchInfoSummaryPath.c_str());
    if (File.is_open()) {
      int Status = std::remove(IndirectBranchInfoSummaryPath.c_str());
      revng_assert(Status == 0);
    }

    std::error_code EC;
    OutputIBI = std::make_unique<raw_fd_ostream>(IndirectBranchInfoSummaryPath,
                                                 EC,
                                                 llvm::sys::fs::OF_Append);
    revng_assert(!EC);

    *OutputIBI << "name,ra,fso,address";
    for (const auto &Reg : ABIRegisters)
      *OutputIBI << "," << Reg->getName();
    *OutputIBI << "\n";
  }

  if (AAWriterPath.getNumOccurrences() == 1) {
    std::ifstream File(AAWriterPath.c_str());
    if (File.is_open()) {
      int Status = std::remove(AAWriterPath.c_str());
      revng_assert(Status == 0);
    }

    std::error_code EC;
    OutputAAWriter = std::make_unique<raw_fd_ostream>(AAWriterPath,
                                                      EC,
                                                      llvm::sys::fs::OF_Append);
    revng_assert(!EC);
  }
}

template<bool FunctionCall>
static model::RegisterState::Values
toRegisterState(RegisterArgument<FunctionCall> RA) {
  switch (RA.value()) {
  case RegisterArgument<FunctionCall>::NoOrDead:
    return model::RegisterState::NoOrDead;
  case RegisterArgument<FunctionCall>::Maybe:
    return model::RegisterState::Maybe;
  case RegisterArgument<FunctionCall>::Yes:
    return model::RegisterState::Yes;
  case RegisterArgument<FunctionCall>::Dead:
    return model::RegisterState::Dead;
  case RegisterArgument<FunctionCall>::Contradiction:
    return model::RegisterState::Contradiction;
  case RegisterArgument<FunctionCall>::No:
    return model::RegisterState::No;
  }

  revng_abort();
}

static model::RegisterState::Values toRegisterState(FunctionReturnValue RV) {
  switch (RV.value()) {
  case FunctionReturnValue::No:
    return model::RegisterState::No;
  case FunctionReturnValue::NoOrDead:
    return model::RegisterState::NoOrDead;
  case FunctionReturnValue::YesOrDead:
    return model::RegisterState::YesOrDead;
  case FunctionReturnValue::Maybe:
    return model::RegisterState::Maybe;
  case FunctionReturnValue::Contradiction:
    return model::RegisterState::Contradiction;
  }

  revng_abort();
}

static model::RegisterState::Values
toRegisterState(FunctionCallReturnValue RV) {
  switch (RV.value()) {
  case FunctionCallReturnValue::No:
    return model::RegisterState::No;
  case FunctionCallReturnValue::NoOrDead:
    return model::RegisterState::NoOrDead;
  case FunctionCallReturnValue::YesOrDead:
    return model::RegisterState::YesOrDead;
  case FunctionCallReturnValue::Yes:
    return model::RegisterState::Yes;
  case FunctionCallReturnValue::Dead:
    return model::RegisterState::Dead;
  case FunctionCallReturnValue::Maybe:
    return model::RegisterState::Maybe;
  case FunctionCallReturnValue::Contradiction:
    return model::RegisterState::Contradiction;
  }

  revng_abort();
}

void commitToModel(GeneratedCodeBasicInfo &GCBI,
                   Function *F,
                   const FunctionsSummary &Summary,
                   model::Binary &TheBinary);

void commitToModel(GeneratedCodeBasicInfo &GCBI,
                   Function *F,
                   const FunctionsSummary &Summary,
                   model::Binary &TheBinary) {
  using namespace model;

  //
  // Create all the model::Function
  //
  for (const auto &[Entry, FunctionSummary] : Summary.Functions) {
    if (Entry == nullptr)
      continue;

    // Get the entry point address
    MetaAddress EntryPC = getBasicBlockPC(Entry);
    revng_assert(EntryPC.isValid());

    model::Function &Function = Binary.Functions[EntryPC];

    // Assign a name

    using FT = model::FunctionType::Values;
    Function.Type = static_cast<FT>(FunctionSummary.Type);

    if (Function.Type == model::FunctionType::Fake)
      continue;

    // Build the function prototype
    auto NewType = makeType<RawFunctionType>();
    auto &FunctionType = *llvm::cast<RawFunctionType>(NewType.get());
    {
      auto ArgumentsInserter = FunctionType.Arguments.batch_insert();
      auto ReturnValuesInserter = FunctionType.ReturnValues.batch_insert();
      for (auto &[CSV, FRD] : FunctionSummary.RegisterSlots) {
        auto RegisterID = ABIRegister::fromCSVName(CSV->getName(), GCBI.arch());
        if (RegisterID == Register::Invalid or CSV == GCBI.spReg())
          continue;

        llvm::Type *CSVType = CSV->getType()->getPointerElementType();
        auto CSVSize = CSVType->getIntegerBitWidth() / 8;
        NamedTypedRegister TR(RegisterID);
        TR.Type = {
          TheBinary.getPrimitiveType(PrimitiveTypeKind::Generic, CSVSize), {}
        };

        if (model::RegisterState::shouldEmit(toRegisterState(FRD.Argument)))
          ArgumentsInserter.insert(TR);

        if (model::RegisterState::shouldEmit(toRegisterState(FRD.ReturnValue)))
          ReturnValuesInserter.insert(TR);

        // TODO: populate preserved registers
      }
    }

    Function.Prototype = TheBinary.recordNewType(std::move(NewType));
  }

  //
  // Populate the CFG
  //
  for (const auto &[Entry, FunctionSummary] : Summary.Functions) {
    if (Entry == nullptr)
      continue;
    MetaAddress EntryPC = getBasicBlockPC(Entry);

    auto It = TheBinary.Functions.find(EntryPC);
    if (It == TheBinary.Functions.end())
      continue;

    model::Function &Function = *It;

    if (Function.Type == model::FunctionType::Fake)
      continue;

    auto MakeEdge = [](MetaAddress Destination, FunctionEdgeType::Values Type) {
      FunctionEdge *Result = nullptr;
      if (FunctionEdgeType::isCall(Type))
        Result = new CallEdge(Destination, Type);
      else
        Result = new FunctionEdge(Destination, Type);
      return UpcastablePointer<FunctionEdge>(Result);
    };

    // Handle the situation in which we found no basic blocks at all
    if (Function.Type == model::FunctionType::NoReturn
        and FunctionSummary.BasicBlocks.size() == 0) {
      auto &EntryNodeSuccessors = Function.CFG[EntryPC].Successors;
      auto Edge = MakeEdge(MetaAddress::invalid(), FunctionEdgeType::LongJmp);
      EntryNodeSuccessors.insert(Edge);
    }

    for (auto &[BB, Branch] : FunctionSummary.BasicBlocks) {
      // Remap BranchType to FunctionEdgeType
      namespace FET = FunctionEdgeType;
      FET::Values EdgeType = FET::Invalid;

      switch (Branch) {
      case BranchType::Invalid:
      case BranchType::FakeFunction:
      case BranchType::RegularFunction:
      case BranchType::NoReturnFunction:
      case BranchType::UnhandledCall:
        revng_abort();
        break;

      case BranchType::InstructionLocalCFG:
        continue;

      default:
        break;
      }

      // Identify Source address
      auto [Source, Size] = getPC(BB->getTerminator());
      Source += Size;
      revng_assert(Source.isValid());

      // Identify Destination address
      llvm::BasicBlock *JumpTargetBB = GCBI.getJumpTargetBlock(BB);
      MetaAddress JumpTargetAddress = GCBI.getPCFromNewPC(JumpTargetBB);
      model::BasicBlock &CurrentBlock = Function.CFG[JumpTargetAddress];
      CurrentBlock.End = Source;
      auto SuccessorsInserter = CurrentBlock.Successors.batch_insert();

      llvm::BasicBlock *Successor = BB->getSingleSuccessor();
      llvm::StringRef SymbolName;

      MetaAddress Destination = MetaAddress::invalid();
      if (Successor != nullptr)
        Destination = getBasicBlockPC(Successor);

      if (Destination.isValid()) {
        revng_assert(not JumpTargetBB->empty());
        auto *NewPCCall = getCallTo(&*Successor->begin(), "newpc");
        revng_assert(NewPCCall != nullptr);

        // Extract symbol name if any
        auto *SymbolNameValue = NewPCCall->getArgOperand(4);
        if (not isa<llvm::ConstantPointerNull>(SymbolNameValue)) {
          llvm::Value *SymbolNameString = NewPCCall->getArgOperand(4);
          SymbolName = extractFromConstantStringPtr(SymbolNameString);
          revng_assert(SymbolName.size() != 0);
        }
      }

      switch (Branch) {
      case BranchType::Invalid:
      case BranchType::FakeFunction:
      case BranchType::RegularFunction:
      case BranchType::NoReturnFunction:
      case BranchType::UnhandledCall:
      case BranchType::InstructionLocalCFG:
        revng_abort();
        break;

      case BranchType::FunctionLocalCFG:
        EdgeType = FET::DirectBranch;
        break;

      case BranchType::FakeFunctionCall:
        EdgeType = FET::FakeFunctionCall;
        break;

      case BranchType::FakeFunctionReturn:
        EdgeType = FET::FakeFunctionReturn;
        break;

      case BranchType::HandledCall:
        EdgeType = FET::FunctionCall;
        break;

      case BranchType::IndirectCall:
        if (SymbolName.size() == 0)
          EdgeType = FET::IndirectCall;
        else
          EdgeType = FET::FunctionCall;
        break;

      case BranchType::Return:
        EdgeType = FET::Return;
        break;

      case BranchType::BrokenReturn:
        EdgeType = FET::BrokenReturn;
        break;

      case BranchType::IndirectTailCall:
        EdgeType = FET::IndirectTailCall;
        break;

      case BranchType::LongJmp:
        EdgeType = FET::LongJmp;
        break;

      case BranchType::Killer:
        EdgeType = FET::Killer;
        break;

      case BranchType::Unreachable:
        EdgeType = FET::Unreachable;
        break;
      }

      if (EdgeType == FET::DirectBranch) {
        // Handle direct branch
        auto Successors = GCBI.getSuccessors(BB);
        for (const MetaAddress &Destination : Successors.Addresses)
          SuccessorsInserter.insert(MakeEdge(Destination, EdgeType));

      } else if (EdgeType == FET::FakeFunctionReturn) {
        // Handle fake function return
        auto [First, Last] = FunctionSummary.FakeReturns.equal_range(BB);
        revng_assert(First != Last);
        for (const auto &[_, Destination] : make_range(First, Last))
          SuccessorsInserter.insert(MakeEdge(Destination, EdgeType));

      } else if (FunctionEdgeType::isCall(EdgeType)) {
        // Record the edge in the CFG
        auto TempEdge = MakeEdge(Destination, EdgeType);
        const auto &Result = SuccessorsInserter.insert(TempEdge);
        auto *Edge = llvm::cast<CallEdge>(Result.get());

        const auto IDF = TheBinary.ImportedDynamicFunctions;
        bool IsDynamicCall = (not SymbolName.empty()
                              and IDF.count(SymbolName.str()) != 0);
        if (IsDynamicCall) {
          // It's a dynamic function call
          revng_assert(EdgeType == model::FunctionEdgeType::FunctionCall);
          Edge->Destination = MetaAddress::invalid();
          Edge->DynamicFunction = SymbolName.str();

          // The prototype is implicitly the one of the callee
          revng_assert(not Edge->Prototype.isValid());
        } else if (Destination.isValid()) {
          // It's a simple direct function call
          revng_assert(EdgeType == model::FunctionEdgeType::FunctionCall);

          // The prototype is implicitly the one of the callee
          revng_assert(not Edge->Prototype.isValid());
        } else {
          // It's an indirect call: forge a new prototype
          auto NewType = makeType<RawFunctionType>();
          auto &CallType = *llvm::cast<RawFunctionType>(NewType.get());
          {
            auto ArgumentsInserter = CallType.Arguments.batch_insert();
            auto ReturnValuesInserter = CallType.ReturnValues.batch_insert();
            bool Found = false;
            for (const FunctionsSummary::CallSiteDescription &CSD :
                 FunctionSummary.CallSites) {
              if (not CSD.Call->isTerminator() or CSD.Call->getParent() != BB)
                continue;

              revng_assert(not Found);
              Found = true;
              for (auto &[CSV, FCRD] : CSD.RegisterSlots) {
                auto RegisterID = ABIRegister::fromCSVName(CSV->getName(),
                                                           GCBI.arch());
                if (RegisterID == model::Register::Invalid
                    or CSV == GCBI.spReg())
                  continue;

                llvm::Type *CSVType = CSV->getType()->getPointerElementType();
                auto CSVSize = CSVType->getIntegerBitWidth() / 8;
                NamedTypedRegister TR(RegisterID);
                TR.Type = {
                  TheBinary.getPrimitiveType(model::PrimitiveTypeKind::Generic,
                                             CSVSize),
                  {}
                };

                auto ArgumentState = toRegisterState(FCRD.Argument);
                if (model::RegisterState::shouldEmit(ArgumentState))
                  ArgumentsInserter.insert(TR);

                auto ReturnValueState = toRegisterState(FCRD.ReturnValue);
                if (model::RegisterState::shouldEmit(ReturnValueState))
                  ReturnValuesInserter.insert(TR);

                // TODO: populate preserved registers and FinalStackOffset
              }
            }
            revng_assert(Found);
          }

          Edge->Prototype = TheBinary.recordNewType(std::move(NewType));
        }
      } else {
        // Handle other successors
        llvm::BasicBlock *Successor = BB->getSingleSuccessor();

        // Record the edge in the CFG
        SuccessorsInserter.insert(MakeEdge(Destination, EdgeType));
      }
    }
  }

  revng_check(TheBinary.verify(true));
}

static void
combineCrossCallSites(MetaAddress EntryPC, FunctionSummary &Summary) {
  using namespace ABIAnalyses;
  using RegState = model::RegisterState::Values;

  for (auto &[PC, CallSites] : Summary.ABIResults.CallSites) {
    if (PC == EntryPC) {
      for (auto &[FuncArg, CSArg] :
           zipmap_range(Summary.ABIResults.ArgumentsRegisters,
                        CallSites.ArgumentsRegisters)) {
        auto *CSV = FuncArg == nullptr ? CSArg->first : FuncArg->first;
        auto RSFArg = FuncArg == nullptr ? RegState::Maybe : FuncArg->second;
        auto RSCSArg = CSArg == nullptr ? RegState::Maybe : CSArg->second;

        Summary.ABIResults.ArgumentsRegisters[CSV] = combine(RSFArg, RSCSArg);
      }
    }
  }
}

/// Perform cross-call site propagation
static void interproceduralPropagation(const std::vector<CFEP> &Functions,
                                       FunctionAnalysisResults &Properties) {
  for (auto &J : Functions) {
    auto CurrentEntryPC = getBasicBlockPC(J.Entry);
    for (auto &K : Functions) {
      auto &Summary = Properties.at(getBasicBlockPC(K.Entry));

      combineCrossCallSites(CurrentEntryPC, Summary);
    }
  }
}

/// Elect a final stack offset to tell whether the function is leaving
/// the stack pointer higher than it was at the function entry.
static std::optional<int64_t> electFSO(const auto &MaybeReturns) {
  auto It = std::min_element(MaybeReturns.begin(),
                             MaybeReturns.end(),
                             [](const auto &LHS, const auto &RHS) {
                               return LHS.second < RHS.second;
                             });
  if (It == MaybeReturns.end())
    return {};
  return It->second;
}

static UpcastablePointer<model::FunctionEdge>
makeEdge(MetaAddress Destination, model::FunctionEdgeType::Values Type) {
  model::FunctionEdge *Result = nullptr;
  using ReturnType = UpcastablePointer<model::FunctionEdge>;

  if (model::FunctionEdgeType::isCall(Type))
    return ReturnType::make<model::CallEdge>(Destination, Type);
  else
    return ReturnType::make<model::FunctionEdge>(Destination, Type);
};

static MetaAddress getFinalAddressOfBasicBlock(llvm::BasicBlock *BB) {
  auto [End, Size] = getPC(BB->getTerminator());
  return End + Size;
}

template<class FunctionOracle>
SortedVector<model::BasicBlock>
CFEPAnalyzer<FunctionOracle>::collectDirectCFG(OutlinedFunction *F) {
  using namespace llvm;

  SortedVector<model::BasicBlock> CFG;

  for (BasicBlock &BB : *F->F) {
    if (GCBI::isJumpTarget(&BB)) {
      MetaAddress Start = getBasicBlockPC(&BB);
      model::BasicBlock Block{ Start };
      Block.End = getFinalAddressOfBasicBlock(&BB);

      OnceQueue<BasicBlock *> Queue;
      Queue.insert(&BB);

      // A JT with no successors?
      if (isa<UnreachableInst>(BB.getTerminator())) {
        auto Type = model::FunctionEdgeType::Unreachable;
        Block.Successors.insert(makeEdge(MetaAddress::invalid(), Type));
      }

      while (!Queue.empty()) {
        BasicBlock *Current = Queue.pop();

        MetaAddress CurrentBlockEnd = getFinalAddressOfBasicBlock(Current);
        if (CurrentBlockEnd > Block.End)
          Block.End = CurrentBlockEnd;

        for (BasicBlock *Succ : successors(Current)) {
          if (GCBI::isJumpTarget(Succ)) {
            MetaAddress Destination = getBasicBlockPC(Succ);
            auto Edge = makeEdge(Destination,
                                 model::FunctionEdgeType::DirectBranch);
            Block.Successors.insert(Edge);
          } else if (F->UnexpectedPCCloned == Succ && succ_size(Current) == 1) {
            // Need to create an edge only when `unexpectedpc` is the unique
            // successor of the current basic block.
            Block.Successors.insert(makeEdge(MetaAddress::invalid(),
                                             model::FunctionEdgeType::LongJmp));
          } else {
            Instruction *I = &(*Succ->begin());

            if (auto *Call = getCallTo(I, PreHookMarker.F)) {
              MetaAddress Destination;
              model::FunctionEdgeType::Values Type;
              auto *CalleePC = Call->getArgOperand(1);

              // Direct or indirect call?
              if (isa<ConstantStruct>(CalleePC)) {
                Destination = MetaAddress::fromConstant(CalleePC);
                Type = model::FunctionEdgeType::FunctionCall;
              } else {
                Destination = MetaAddress::invalid();
                Type = model::FunctionEdgeType::IndirectCall;
              }

              auto Edge = makeEdge(Destination, Type);
              auto DestTy = Oracle.getFunctionType(Destination);
              if (DestTy == FunctionTypeValue::NoReturn) {
                auto *CE = cast<model::CallEdge>(Edge.get());
                CE->Attributes.insert(model::FunctionAttribute::NoReturn);
              }

              Block.Successors.insert(Edge);
            } else if (auto *Call = getCallTo(I, "function_call")) {
              // At this stage, `function_call` marker has only been left to
              // signal the presence of fake functions. We can safely erase it
              // and add an edge of type FakeFunctionCall (still used in
              // IsolateFunction).
              Call->eraseFromParent();

              auto Destination = getBasicBlockPC(Succ);
              auto Edge = makeEdge(Destination,
                                   model::FunctionEdgeType::FakeFunctionCall);
              Block.Successors.insert(Edge);
            } else {
              Queue.insert(Succ);
            }
          }
        }
      }

      CFG.insert(Block);
    }
  }

  return CFG;
}

template<class FO>
void CFEPAnalyzer<FO>::initMarkersForABI(OutlinedFunction *OutlinedFunction,
                                         SmallVectorImpl<Instruction *> &SV,
                                         llvm::IRBuilder<> &IRB) {
  using namespace llvm;

  StructType *MetaAddressTy = MetaAddress::getStruct(&M);
  SmallVector<Instruction *, 4> IndirectBranchPredecessors;
  if (OutlinedFunction->AnyPCCloned) {
    for (auto *Pred : predecessors(OutlinedFunction->AnyPCCloned)) {
      auto *Term = Pred->getTerminator();
      IndirectBranchPredecessors.emplace_back(Term);
    }
  }

  // Initialize ret-hook marker (needed for the ABI analyses on return values)
  // and fix pre-hook marker upon encountering a jump to `anypc`. Since we don't
  // know in advance whether it will be classified as a return or indirect tail
  // call, ABIAnalyses (e.g., RAOFC) need to run on this potential call-site as
  // well. The results will not be merged eventually, if the indirect jump turns
  // out to be a proper return.
  for (auto *Term : IndirectBranchPredecessors) {
    auto *BB = Term->getParent();

    MetaAddress IndirectRetBBAddress = GCBI->getJumpTarget(BB);
    revng_assert(IndirectRetBBAddress.isValid());

    auto *Split = BB->splitBasicBlock(Term, BB->getName() + Twine("_anypc"));
    auto *JumpToAnyPC = Split->getTerminator();
    revng_assert(isa<BranchInst>(JumpToAnyPC));
    IRB.SetInsertPoint(JumpToAnyPC);

    IRB.CreateCall(PreHookMarker.F,
                   { IndirectRetBBAddress.toConstant(MetaAddressTy),
                     MetaAddress::invalid().toConstant(MetaAddressTy) });

    IRB.CreateCall(RetHookMarker.F,
                   { IndirectRetBBAddress.toConstant(MetaAddressTy),
                     MetaAddress::invalid().toConstant(MetaAddressTy) });

    SV.emplace_back(JumpToAnyPC);
  }
}

template<class FO>
std::set<llvm::GlobalVariable *>
CFEPAnalyzer<FO>::findWrittenRegisters(llvm::Function *F) {
  using namespace llvm;

  std::set<GlobalVariable *> WrittenRegisters;
  for (auto &BB : *F) {
    for (auto &I : BB) {
      if (auto *SI = dyn_cast<StoreInst>(&I)) {
        Value *Ptr = skipCasts(SI->getPointerOperand());
        if (auto *GV = dyn_cast<GlobalVariable>(Ptr))
          WrittenRegisters.insert(GV);
      }
    }
  }

  return WrittenRegisters;
}

template<class FO>
void CFEPAnalyzer<FO>::createIBIMarker(OutlinedFunction *OutlinedFunction,
                                       SmallVectorImpl<Instruction *> &SV,
                                       llvm::IRBuilder<> &IRB) {
  using namespace llvm;

  StructType *MetaAddressTy = MetaAddress::getStruct(&M);
  IRB.SetInsertPoint(&OutlinedFunction->F->getEntryBlock().front());
  auto *IntPtrTy = GCBI->spReg()->getType();
  auto *IntTy = GCBI->spReg()->getType()->getElementType();

  // At the entry of the function, load the initial value of stack pointer,
  // program counter and ABI registers used within this function.
  auto *SP = IRB.CreateLoad(GCBI->spReg());
  auto *SPPtr = IRB.CreateIntToPtr(SP, IntPtrTy);
  auto *GEPI = IRB.CreateGEP(IntTy, SPPtr, ConstantInt::get(IntTy, 0));

  Value *RA = IRB.CreateLoad(GCBI->raReg() ? GCBI->raReg() : SPPtr);

  std::array<Value *, 4> DissectedPC = PCH->dissectJumpablePC(IRB,
                                                              RA,
                                                              GCBI->arch());

  auto *PCI = MetaAddress::composeIntegerPC(IRB,
                                            DissectedPC[0],
                                            DissectedPC[1],
                                            DissectedPC[2],
                                            DissectedPC[3]);

  SmallVector<Value *, 16> CSVI;
  Type *IsRetTy = Type::getInt128Ty(Context);
  SmallVector<Type *, 16> ArgTypes = { IsRetTy, IntTy, MetaAddressTy };
  for (auto *CSR : ABIRegisters) {
    auto *V = IRB.CreateLoad(CSR);
    CSVI.emplace_back(V);
    ArgTypes.emplace_back(IntTy);
  }

  auto *FTy = llvm::FunctionType::get(IntTy, ArgTypes, false);
  OutlinedFunction
    ->IndirectBranchInfoMarker = Function::Create(FTy,
                                                  GlobalValue::ExternalLinkage,
                                                  "indirect_branch_info",
                                                  M);
  OutlinedFunction->IndirectBranchInfoMarker->addFnAttr(Attribute::NoUnwind);
  OutlinedFunction->IndirectBranchInfoMarker->addFnAttr(Attribute::NoReturn);

  // When an indirect jump is encountered (possible exit point), a dedicated
  // basic block is created, and the values of the stack pointer, program
  // counter and ABI registers are loaded.
  for (auto *Term : SV) {
    auto *IBIBlock = BasicBlock::Create(Context,
                                        Term->getParent()->getName()
                                          + Twine("_indirect_branch_info"),
                                        OutlinedFunction->F,
                                        nullptr);

    Term->replaceUsesOfWith(OutlinedFunction->AnyPCCloned, IBIBlock);

    IRB.SetInsertPoint(IBIBlock);
    auto *PCE = PCH->composeIntegerPC(IRB);

    SmallVector<Value *, 16> CSVE;
    for (auto *CSR : ABIRegisters) {
      auto *V = IRB.CreateLoad(CSR);
      CSVE.emplace_back(V);
    }

    SP = IRB.CreateLoad(GCBI->spReg());
    SPPtr = IRB.CreateIntToPtr(SP, IntPtrTy);
    auto *GEPE = IRB.CreateGEP(IntTy, SPPtr, ConstantInt::get(IntTy, 0));

    auto *SPI = IRB.CreatePtrToInt(GEPI, IntTy);
    auto *SPE = IRB.CreatePtrToInt(GEPE, IntTy);

    // Compute the difference between the program counter at entry and exit
    // function. Should it turn out to be zero, the function jumps to its return
    // address.
    auto *JumpsToReturnAddress = IRB.CreateSub(PCE, PCI);

    // Compute the difference between the stack pointer values to evaluate the
    // stack height. Functions leaving the stack pointer higher than it was at
    // function entry (i.e., in an irregular state) will be marked as fake
    // functions.
    auto *StackPointerDifference = IRB.CreateSub(SPE, SPI);

    // Save the MetaAddress of the final jump target
    auto NewPCJT = GCBI->getJumpTarget(Term->getParent());
    revng_assert(NewPCJT.isValid());

    SmallVector<Value *, 16> ArgValues = { JumpsToReturnAddress,
                                           StackPointerDifference,
                                           NewPCJT.toConstant(MetaAddressTy) };

    // Compute the difference between the initial and final values of the CSV
    // ABI registers. Should it turn out to be zero, the CSV is preserved across
    // the function call (callee-saved).
    for (const auto &[Initial, End] : zip(CSVI, CSVE)) {
      auto *ABIRegistersDifference = IRB.CreateSub(Initial, End);
      ArgValues.emplace_back(ABIRegistersDifference);
    }

    // Install the `indirect_branch_info` call
    IRB.CreateCall(OutlinedFunction->IndirectBranchInfoMarker, ArgValues);
    IRB.CreateUnreachable();
  }
}

template<class FO>
void CFEPAnalyzer<FO>::opaqueBranchConditions(llvm::Function *F,
                                              llvm::IRBuilder<> &IRB) {
  using namespace llvm;

  for (auto &BB : *F) {
    auto *Term = BB.getTerminator();
    if ((isa<BranchInst>(Term) && cast<BranchInst>(Term)->isConditional())
        || isa<SwitchInst>(Term)) {
      Value *Condition = isa<BranchInst>(Term) ?
                           cast<BranchInst>(Term)->getCondition() :
                           cast<SwitchInst>(Term)->getCondition();

      OpaqueBranchConditionsPool.addFnAttribute(Attribute::NoUnwind);
      OpaqueBranchConditionsPool.addFnAttribute(Attribute::ReadOnly);
      OpaqueBranchConditionsPool.addFnAttribute(Attribute::WillReturn);

      auto *FTy = llvm::FunctionType::get(Condition->getType(),
                                          { Condition->getType() },
                                          false);

      auto *OpaqueTrueCallee = OpaqueBranchConditionsPool.get(FTy,
                                                              FTy,
                                                              "opaque_true");

      IRB.SetInsertPoint(Term);
      auto *RetVal = IRB.CreateCall(OpaqueTrueCallee, { Condition });

      if (isa<BranchInst>(Term))
        cast<BranchInst>(Term)->setCondition(RetVal);
      else
        cast<SwitchInst>(Term)->setCondition(RetVal);
    }
  }
}

template<class FO>
void CFEPAnalyzer<FO>::materializePCValues(llvm::Function *F,
                                           llvm::IRBuilder<> &IRB) {
  using namespace llvm;

  for (auto &BB : *F) {
    for (auto &I : BB) {
      if (auto *Call = getCallTo(&I, "newpc")) {
        MetaAddress NewPC = GCBI::getPCFromNewPC(Call);
        IRB.SetInsertPoint(Call);
        PCH->setPC(IRB, NewPC);
      }
    }
  }
}

template<typename T>
struct TemporaryOption {
public:
  TemporaryOption(const char *Name, const T &Value) :
    Name(Name), Options(llvm::cl::getRegisteredOptions()) {
    OldValue = Opt(Options, Name)->getValue();
    Opt(Options, Name)->setInitialValue(Value);
  }

  ~TemporaryOption() { Opt(Options, Name)->setInitialValue(OldValue); }

private:
  T OldValue;
  const char *Name;
  llvm::StringMap<llvm::cl::Option *> &Options;
  static constexpr const auto &Opt = getOption<T>;
};

template<class FO>
void CFEPAnalyzer<FO>::runOptimizationPipeline(llvm::Function *F) {
  using namespace llvm;

  // Some LLVM passes used later in the pipeline scan for cut-offs, meaning that
  // further computation may not be done when they are reached; making some
  // optimizations opportunities missed. Hence, we set the involved thresholds
  // (e.g., the maximum value that MemorySSA uses to take into account
  // stores/phis) to have initial unbounded value.
  static constexpr const char *MemSSALimit = "memssa-check-limit";
  static constexpr const char *MemDepBlockLimit = "memdep-block-scan-limit";

  using TemporaryUOption = TemporaryOption<unsigned>;
  TemporaryUOption MemSSALimitOption(MemSSALimit, UINT_MAX);
  TemporaryUOption MemDepBlockLimitOption(MemDepBlockLimit, UINT_MAX);

  // TODO: break it down in the future, and check if some passes can be dropped
  {
    FunctionPassManager FPM;

    // First stage: simplify the IR, promote the CSVs to local variables,
    // compute subexpressions elimination and resolve redundant expressions in
    // order to compute the stack height.
    FPM.addPass(RemoveNewPCCallsPass());
    FPM.addPass(RemoveHelperCallsPass());
    FPM.addPass(PromoteGlobalToLocalPass());
    FPM.addPass(SimplifyCFGPass());
    FPM.addPass(SROA());
    FPM.addPass(EarlyCSEPass(true));
    FPM.addPass(JumpThreadingPass());
    FPM.addPass(UnreachableBlockElimPass());
    FPM.addPass(InstCombinePass(true));
    FPM.addPass(EarlyCSEPass(true));
    FPM.addPass(SimplifyCFGPass());
    FPM.addPass(MergedLoadStoreMotionPass());
    FPM.addPass(GVN());

    // Second stage: add alias analysis info and canonicalize `i2p` + `add` into
    // `getelementptr` instructions. Since the IR may change remarkably, another
    // round of passes is necessary to take more optimization opportunities.
    FPM.addPass(SegregateDirectStackAccessesPass());
    FPM.addPass(EarlyCSEPass(true));
    FPM.addPass(InstCombinePass(true));
    FPM.addPass(GVN());

    // Third stage: if enabled, serialize the results and dump the functions on
    // disk with the alias information included as comments.
    if (IndirectBranchInfoSummaryPath.getNumOccurrences() == 1)
      FPM.addPass(IndirectBranchInfoPrinterPass(*OutputIBI));

    if (AAWriterPath.getNumOccurrences() == 1)
      FPM.addPass(AAWriterPass(*OutputAAWriter));

    ModuleAnalysisManager MAM;

    FunctionAnalysisManager FAM;
    FAM.registerPass([] {
      AAManager AA;
      AA.registerFunctionAnalysis<BasicAA>();
      AA.registerFunctionAnalysis<ScopedNoAliasAA>();

      return AA;
    });
    FAM.registerPass([&] { return GeneratedCodeBasicInfoAnalysis(); });
    FAM.registerPass([&] { return ModuleAnalysisManagerFunctionProxy(MAM); });

    PassBuilder PB;
    PB.registerFunctionAnalyses(FAM);
    PB.registerModuleAnalyses(MAM);

    FPM.run(*F, FAM);
  }
}

template<class FO>
llvm::Function *CFEPAnalyzer<FO>::createFakeFunction(llvm::BasicBlock *Entry) {
  using namespace llvm;

  // Recreate outlined function
  OutlinedFunction FakeFunction = outlineFunction(Entry);

  // Adjust `anypc` and `unexpectedpc` BBs of the fake function
  revng_assert(FakeFunction.AnyPCCloned != nullptr);

  // Fake functions must have one and only one broken return
  revng_assert(FakeFunction.AnyPCCloned->hasNPredecessors(1));

  // Replace the broken return with a `ret`
  auto *Br = FakeFunction.AnyPCCloned->getUniquePredecessor()->getTerminator();
  auto *Ret = ReturnInst::Create(Context);
  ReplaceInstWithInst(Br, Ret);

  if (FakeFunction.UnexpectedPCCloned != nullptr) {
    CallInst::Create(UnexpectedPCMarker.F,
                     "",
                     FakeFunction.UnexpectedPCCloned->getTerminator());
  }

  return FakeFunction.extractFunction();
}

template<class FO>
FunctionSummary CFEPAnalyzer<FO>::analyze(BasicBlock *Entry) {
  using namespace llvm;
  using namespace ABIAnalyses;

  IRBuilder<> Builder(M.getContext());
  SmallVector<Instruction *, 4> BranchesForIBI;

  // Detect function boundaries
  struct OutlinedFunction OutlinedFunction = outlineFunction(Entry);

  // Recover the control-flow graph of the function
  auto CFG = collectDirectCFG(&OutlinedFunction);

  // Initalize markers for ABI analyses and set up the branches on which
  // `indirect_branch_info` will be installed.
  initMarkersForABI(&OutlinedFunction, BranchesForIBI, Builder);

  // Find registers that may be target of at least one store. This helps
  // refine the final results.
  auto WrittenRegisters = findWrittenRegisters(OutlinedFunction.F);

  // Run ABI-independent data-flow analyses
  ABIAnalysesResults ABIResults = analyzeOutlinedFunction(OutlinedFunction.F,
                                                          *GCBI,
                                                          PreHookMarker.F,
                                                          PostHookMarker.F,
                                                          RetHookMarker.F);

  // Recompute the DomTree for the current outlined function due to split basic
  // blocks.
  GCBI->purgeDomTree(OutlinedFunction.F);

  // The analysis aims at identifying the callee-saved registers of a function
  // and establishing if a function returns properly, i.e., it jumps to the
  // return address (regular function). In order to achieve this, the IR is
  // crafted by loading the program counter, the stack pointer, as well as the
  // ABI registers CSVs respectively at function prologue / epilogue. When the
  // subtraction between their entry and end values is found to be zero (after
  // running an LLVM optimization pipeline), we may infer if the function
  // returns correctly, the stack is left unanaltered, etc. Hence, upon every
  // original indirect jump (candidate exit point), a marker of this kind is
  // installed:
  //
  //                           jumps to RA,    SP,    rax,   rbx,   rbp
  // call i64 @indirect_branch_info(i128 0, i64 8, i64 %8, i64 0, i64 0)
  //
  // Here, subsequently the opt pipeline computation, we may tell that the
  // function jumps to its return address (thus, it is not a longjmp / tail
  // call), `rax` register has been clobbered by the callee, whereas `rbx` and
  // `rbp` are callee-saved registers.
  createIBIMarker(&OutlinedFunction, BranchesForIBI, Builder);

  // Prevent DCE by making branch conditions opaque
  opaqueBranchConditions(OutlinedFunction.F, Builder);

  // Store the values that build up the program counter in order to have them
  // constant-folded away by the optimization pipeline.
  materializePCValues(OutlinedFunction.F, Builder);

  // Execute the optimization pipeline over the outlined function
  runOptimizationPipeline(OutlinedFunction.F);

  // Squeeze out the results obtained from the optimization passes
  auto FunctionInfo = milkInfo(&OutlinedFunction,
                               CFG,
                               ABIResults,
                               WrittenRegisters);

  // Does the outlined function basically represent a function prologue? If so,
  // the function is said to be fake, and a copy of the unoptimized outlined
  // function is returned. When analyzing the caller, this function will be
  // inlined in its call-site.
  if (FunctionInfo.Type == FunctionTypeValue::Fake)
    FunctionInfo.FakeFunction = createFakeFunction(Entry);

  // Reset the DomTree for the current outlined function
  GCBI->purgeDomTree(OutlinedFunction.F);

  return FunctionInfo;
}

static void
suppressCSAndSPRegisters(ABIAnalyses::ABIAnalysesResults &ABIResults,
                         const std::set<GlobalVariable *> &CalleeSavedRegs) {
  using RegisterState = model::RegisterState::Values;

  // Suppress from arguments
  for (const auto &Reg : CalleeSavedRegs) {
    auto It = ABIResults.ArgumentsRegisters.find(Reg);
    if (It != ABIResults.ArgumentsRegisters.end())
      It->second = RegisterState::No;
  }

  // Suppress from return values
  for (const auto &[K, _] : ABIResults.ReturnValuesRegisters) {
    for (const auto &Reg : CalleeSavedRegs) {
      auto It = ABIResults.ReturnValuesRegisters[K].find(Reg);
      if (It != ABIResults.ReturnValuesRegisters[K].end())
        It->second = RegisterState::No;
    }
  }

  // Suppress from call-sites
  for (const auto &[K, _] : ABIResults.CallSites) {
    for (const auto &Reg : CalleeSavedRegs) {
      if (ABIResults.CallSites[K].ArgumentsRegisters.count(Reg) != 0)
        ABIResults.CallSites[K].ArgumentsRegisters[Reg] = RegisterState::No;

      if (ABIResults.CallSites[K].ReturnValuesRegisters.count(Reg) != 0)
        ABIResults.CallSites[K].ReturnValuesRegisters[Reg] = RegisterState::No;
    }
  }
}

static void discardBrokenReturns(ABIAnalyses::ABIAnalysesResults &ABIResults,
                                 const auto &IBIResult) {
  for (const auto &[CI, EdgeType] : IBIResult) {
    if (EdgeType != FunctionEdgeTypeValue::Return
        && EdgeType != FunctionEdgeTypeValue::IndirectTailCall) {
      auto PC = MetaAddress::fromConstant(CI->getOperand(2));

      auto It = ABIResults.ReturnValuesRegisters.find(PC);
      if (It != ABIResults.ReturnValuesRegisters.end())
        ABIResults.ReturnValuesRegisters.erase(PC);
    }
  }
}

static std::set<GlobalVariable *>
intersect(const std::set<GlobalVariable *> &First,
          const std::set<GlobalVariable *> &Last) {
  std::set<GlobalVariable *> Output;
  std::set_intersection(First.begin(),
                        First.end(),
                        Last.begin(),
                        Last.end(),
                        std::inserter(Output, Output.begin()));
  return Output;
}

template<class FO>
FunctionSummary
CFEPAnalyzer<FO>::milkInfo(OutlinedFunction *OutlinedFunction,
                           SortedVector<model::BasicBlock> &CFG,
                           ABIAnalyses::ABIAnalysesResults &ABIResults,
                           const std::set<GlobalVariable *> &WrittenRegisters) {
  using namespace llvm;

  SmallVector<std::pair<CallBase *, int64_t>, 4> MaybeReturns;
  SmallVector<std::pair<CallBase *, int64_t>, 4> NotReturns;
  SmallVector<std::pair<CallBase *, FunctionEdgeTypeValue>, 4> IBIResult;
  std::set<GlobalVariable *> CalleeSavedRegs;
  std::set<GlobalVariable *> ClobberedRegs(ABIRegisters.begin(),
                                           ABIRegisters.end());

  for (CallBase *CI : callers(OutlinedFunction->IndirectBranchInfoMarker)) {
    if (CI->getParent()->getParent() == OutlinedFunction->F) {
      bool JumpsToReturnAddress = false;
      auto MayJumpToReturnAddress = dyn_cast<ConstantInt>(CI->getArgOperand(0));
      if (MayJumpToReturnAddress)
        JumpsToReturnAddress = MayJumpToReturnAddress->getSExtValue() == 0;

      auto *StackPointerOffset = dyn_cast<ConstantInt>(CI->getArgOperand(1));
      if (StackPointerOffset) {
        int64_t FSO = StackPointerOffset->getSExtValue();
        if (JumpsToReturnAddress) {
          if (FSO >= 0)
            MaybeReturns.emplace_back(CI, FSO);
          else
            IBIResult.emplace_back(CI, FunctionEdgeTypeValue::BrokenReturn);
        } else {
          NotReturns.emplace_back(CI, FSO);
        }
      } else {
        if (JumpsToReturnAddress)
          IBIResult.emplace_back(CI, FunctionEdgeTypeValue::LongJmp);
      }
    }
  }

  // Elect a final stack offset
  auto WinFSO = electFSO(MaybeReturns);

  // Did we find at least a valid return instruction?
  for (const auto &[CI, FSO] : MaybeReturns) {
    if (FSO == *WinFSO) {
      IBIResult.emplace_back(CI, FunctionEdgeTypeValue::Return);
      unsigned ArgumentsCount = CI->getNumArgOperands();
      if (ArgumentsCount > 3) {
        for (unsigned Idx = 3; Idx < ArgumentsCount; ++Idx) {
          auto *Register = dyn_cast<ConstantInt>(CI->getArgOperand(Idx));
          if (Register && Register->getZExtValue() == 0)
            CalleeSavedRegs.insert(ABIRegisters[Idx - 3]);
        }
      }
    } else {
      IBIResult.emplace_back(CI, FunctionEdgeTypeValue::BrokenReturn);
    }
  }

  // Neither a return nor a broken return? Re-elect a FSO taking into account no
  // returns indirect jumps only.
  if (!WinFSO.has_value())
    WinFSO = electFSO(NotReturns);

  for (CallBase *CI : callers(OutlinedFunction->IndirectBranchInfoMarker)) {
    if (CI->getParent()->getParent() == OutlinedFunction->F) {
      auto MayJumpToReturnAddress = dyn_cast<ConstantInt>(CI->getArgOperand(0));
      if (MayJumpToReturnAddress && MayJumpToReturnAddress->getSExtValue() == 0)
        continue;

      // We have an indirect jump and we classify it depending on the status of
      // the stack pointer.
      auto *StackOffset = dyn_cast<ConstantInt>(CI->getArgOperand(1));
      if (WinFSO.has_value() && StackOffset != nullptr
          && StackOffset->getSExtValue() == *WinFSO)
        IBIResult.emplace_back(CI, FunctionEdgeTypeValue::IndirectTailCall);
      else
        IBIResult.emplace_back(CI, FunctionEdgeTypeValue::LongJmp);
    }
  }

  bool FoundReturn = false;
  bool FoundBrokenReturn = false;
  int BrokenReturnCount = 0, NoReturnCount = 0;
  for (const auto &[CI, EdgeType] : IBIResult) {
    if (EdgeType == FunctionEdgeTypeValue::Return) {
      FoundReturn = true;
    } else if (EdgeType == FunctionEdgeTypeValue::BrokenReturn) {
      FoundBrokenReturn = true;
      BrokenReturnCount++;
    } else {
      NoReturnCount++;
    }
  }

  // Function is elected fake if there is one and only one broken return
  FunctionTypeValue Type;
  if (FoundReturn) {
    Type = FunctionTypeValue::Regular;
  } else if (FoundBrokenReturn && BrokenReturnCount == 1
             && NoReturnCount == 0) {
    Type = FunctionTypeValue::Fake;
  } else {
    Type = FunctionTypeValue::NoReturn;
  }

  // Retrieve the clobbered registers
  std::erase_if(ClobberedRegs,
                [&](const auto &E) { return CalleeSavedRegs.count(E) != 0; });

  // Finalize CFG for the model
  for (const auto &[CI, EdgeType] : IBIResult) {
    auto PC = MetaAddress::fromConstant(CI->getArgOperand(2));
    model::BasicBlock &Block = CFG.at(PC);
    Block.Successors.insert(makeEdge(MetaAddress::invalid(), EdgeType));
  }

  // Empty CFG if function is fake
  if (Type == FunctionTypeValue::Fake)
    CFG.clear();

  // We say that a register is callee-saved when, besides being preserved by the
  // callee, there is at least a write onto this register.
  auto ActualCalleeSavedRegs = intersect(CalleeSavedRegs, WrittenRegisters);

  // Union between effective callee-saved registers and SP
  ActualCalleeSavedRegs.insert(GCBI->spReg());

  // Refine ABI analyses results by suppressing callee-saved and stack pointer
  // registers.
  suppressCSAndSPRegisters(ABIResults, ActualCalleeSavedRegs);

  // ABI analyses run from all the `indirect_branch_info` (i.e., all candidate
  // returns). We are going to merge the results only from those return points
  // that have been classified as proper return (i.e., no broken return).
  discardBrokenReturns(ABIResults, IBIResult);

  // Merge return values registers
  ABIAnalyses::finalizeReturnValues(ABIResults);

  ABIResults.dump(StackAnalysisLog);

  return FunctionSummary(Type,
                         std::move(ClobberedRegs),
                         std::move(ABIResults),
                         std::move(CFG),
                         WinFSO,
                         nullptr);
}

template<class FunctionOracle>
void CFEPAnalyzer<FunctionOracle>::integrateFunctionCallee(llvm::BasicBlock *BB,
                                                           MetaAddress Next) {
  using namespace llvm;

  // If the basic block originally had a call-site, the function call is
  // replaced with 1) hooks that delimit the space of the ABI analyses'
  // traversals and 2) a summary of the registers clobbered by that function.
  auto *Term = BB->getTerminator();
  auto *Call = getFunctionCall(Term);

  // What is the function type of the callee?
  FunctionTypeValue Type = Oracle.getFunctionType(Next);

  switch (Type) {
  case FunctionTypeValue::Regular:
  case FunctionTypeValue::NoReturn: {
    // Extract MetaAddress of JT of the call-site
    auto CallSiteJT = GCBI->getJumpTarget(BB);
    revng_assert(CallSiteJT.isValid());

    // What are the registers clobbered by the callee?
    const auto &ClobberedRegisters = Oracle.getRegistersClobbered(Next);

    // Different insert point depending on the callee type
    IRBuilder<> Builder(M.getContext());
    if (Type == FunctionTypeValue::Regular) {
      Builder.SetInsertPoint(Term);
    } else {
      auto *AbortCall = dyn_cast<CallInst>(Term->getPrevNode());
      revng_assert(AbortCall != nullptr
                   && AbortCall->getCalledFunction() == M.getFunction("abort"));
      Builder.SetInsertPoint(AbortCall);
    }

    // Mark end of basic block with a pre-hook call
    StructType *MetaAddressTy = MetaAddress::getStruct(&M);
    SmallVector<Value *, 2> Args = { CallSiteJT.toConstant(MetaAddressTy),
                                     Next.toConstant(MetaAddressTy) };
    auto *Last = Builder.CreateCall(PreHookMarker.F, Args);

    // Prevent the store instructions from being optimized out by storing
    // the rval of a call to an opaque function into the clobbered registers.
    RegistersClobberedPool.addFnAttribute(Attribute::ReadOnly);
    RegistersClobberedPool.addFnAttribute(Attribute::NoUnwind);
    RegistersClobberedPool.addFnAttribute(Attribute::WillReturn);

    for (GlobalVariable *Register : ClobberedRegisters) {
      auto *CSVTy = Register->getType()->getPointerElementType();
      auto Name = ("registers_clobbered_" + Twine(Register->getName())).str();
      auto *OpaqueRegistersClobberedCallee = RegistersClobberedPool
                                               .get(Register->getName(),
                                                    CSVTy,
                                                    {},
                                                    Name);

      Builder.CreateStore(Builder.CreateCall(OpaqueRegistersClobberedCallee),
                          Register);
    }

    // Adjust back the stack pointer
    switch (GCBI->arch()) {
    case llvm::Triple::x86:
    case llvm::Triple::x86_64: {
      auto *SP = Builder.CreateLoad(GCBI->spReg());

      const auto &FSO = Oracle.getElectedFSO(Next);
      Value *Offset = GCBI->arch() == llvm::Triple::x86 ?
                        Builder.getInt32(*FSO) :
                        Builder.getInt64(*FSO);
      auto *Inc = Builder.CreateAdd(SP, Offset);
      Builder.CreateStore(Inc, GCBI->spReg());
      break;
    }
    default: {
      break;
    }
    }

    // Mark end of basic block with a post-hook call
    Builder.CreateCall(PostHookMarker.F, Args);

    BB->splitBasicBlock(Last->getPrevNode(),
                        BB->getName() + Twine("__summary"));

    // Erase the `function_call` marker unless fake
    Call->eraseFromParent();
    break;
  }

  case FunctionTypeValue::Fake: {
    // Get the fake function by its entry basic block
    Function *FakeFunction = Oracle.getFakeFunction(Next);

    // If fake, it must have been already analyzed
    revng_assert(FakeFunction != nullptr);

    // If possible, inline the fake function
    auto *CI = CallInst::Create(FakeFunction, "", Term);
    InlineFunctionInfo IFI;
    bool Status = InlineFunction(*CI, IFI, nullptr, true).isSuccess();
    revng_log(StackAnalysisLog,
              "Has callee " << FakeFunction->getName() << "been inlined? "
                            << Status);
    break;
  }

  default:
    revng_abort();
  }
}

template<class FO>
OutlinedFunction CFEPAnalyzer<FO>::outlineFunction(llvm::BasicBlock *Entry) {
  using namespace llvm;

  Function *Root = Entry->getParent();

  OutlinedFunction OutlinedFunction;
  OnceQueue<BasicBlock *> Queue;
  std::vector<BasicBlock *> BlocksToClone;
  Queue.insert(Entry);

  // Collect list of blocks to clone
  while (!Queue.empty()) {
    BasicBlock *Current = Queue.pop();
    BlocksToClone.emplace_back(Current);

    if (isFunctionCall(Current)) {
      auto *Successor = getFallthrough(Current);
      MetaAddress PCCallee = MetaAddress::invalid();
      if (auto *Next = getFunctionCallCallee(Current))
        PCCallee = getBasicBlockPC(Next);

      if (Oracle.getFunctionType(PCCallee) != FunctionTypeValue::NoReturn)
        Queue.insert(Successor);
    } else {
      for (auto *Successor : successors(Current)) {
        if (!GCBI::isPartOfRootDispatcher(Successor))
          Queue.insert(Successor);
      }
    }
  }

  // Create a copy of all the basic blocks to outline in `root`
  ValueToValueMapTy VMap;
  SmallVector<BasicBlock *, 8> BlocksToExtract;

  for (const auto &BB : BlocksToClone) {
    BasicBlock *Cloned = CloneBasicBlock(BB, VMap, Twine("_cloned"), Root);

    VMap[BB] = Cloned;
    BlocksToExtract.emplace_back(Cloned);
  }

  auto AnyPCIt = VMap.find(GCBI->anyPC());
  if (AnyPCIt != VMap.end())
    OutlinedFunction.AnyPCCloned = cast<BasicBlock>(AnyPCIt->second);

  auto UnexpPCIt = VMap.find(GCBI->unexpectedPC());
  if (UnexpPCIt != VMap.end())
    OutlinedFunction.UnexpectedPCCloned = cast<BasicBlock>(UnexpPCIt->second);

  remapInstructionsInBlocks(BlocksToExtract, VMap);

  // Fix successor when encountering a call-site and fix fall-through in
  // presence of a noreturn function.
  std::map<llvm::CallInst *, MetaAddress> CallMap;
  for (const auto &BB : BlocksToExtract) {
    if (isFunctionCall(BB)) {
      auto *Term = BB->getTerminator();
      CallInst *CI = getFunctionCall(Term);

      // If the function callee is null, we are dealing with an indirect call
      MetaAddress PCCallee = MetaAddress::invalid();
      if (BasicBlock *Next = getFunctionCallCallee(Term))
        PCCallee = getBasicBlockPC(Next);

      auto CalleeType = Oracle.getFunctionType(PCCallee);

      if (CalleeType != FunctionTypeValue::NoReturn) {
        auto *Br = BranchInst::Create(getFallthrough(Term));
        ReplaceInstWithInst(Term, Br);
      } else if (CalleeType == FunctionTypeValue::NoReturn) {
        auto *Abort = CallInst::Create(M.getFunction("abort"));
        new UnreachableInst(Term->getContext(), BB);
        ReplaceInstWithInst(Term, Abort);
      }

      // To allow correct function extraction, there must not exist users of BBs
      // to be extracted, so we destroy the blockaddress of the fall-through BB
      // in the `function_call` marker.
      unsigned ArgNo = 0;
      PointerType *I8PtrTy = Type::getInt8PtrTy(M.getContext());
      Constant *I8NullPtr = ConstantPointerNull::get(I8PtrTy);
      for (Value *Arg : CI->args()) {
        if (isa<BlockAddress>(Arg)) {
          CI->setArgOperand(ArgNo, I8NullPtr);
          if (Arg->use_empty())
            cast<BlockAddress>(Arg)->destroyConstant();
        }
        ++ArgNo;
      }

      CallMap.insert({ CI, PCCallee });
    }
  }

  // Extract outlined function
  OutlinedFunction.F = CodeExtractor(BlocksToExtract).extractCodeRegion(CEAC);

  revng_assert(OutlinedFunction.F != nullptr);
  revng_assert(OutlinedFunction.F->arg_size() == 0);
  revng_assert(OutlinedFunction.F->getReturnType()->isVoidTy());
  revng_assert(OutlinedFunction.F->hasOneUser());

  // Remove the only user (call to the outlined function) in `root`
  auto It = OutlinedFunction.F->user_begin();
  cast<Instruction>(*It)->getParent()->eraseFromParent();

  // Integrate function callee
  for (auto &BB : *OutlinedFunction.F) {
    if (isFunctionCall(&BB)) {
      auto *Term = BB.getTerminator();
      MetaAddress CalleePC = CallMap.at(getFunctionCall(Term));
      integrateFunctionCallee(&BB, CalleePC);
    }
  }

  // TODO: fix `unexpectedpc` of the fake callee
  for (auto &I : instructions(OutlinedFunction.F)) {
    if (CallInst *Call = getCallTo(&I, UnexpectedPCMarker.F)) {
      // TODO: can `unexpectedpc` not exist in the caller?
      revng_assert(OutlinedFunction.UnexpectedPCCloned != nullptr);

      auto *Br = BranchInst::Create(OutlinedFunction.UnexpectedPCCloned);
      ReplaceInstWithInst(I.getParent()->getTerminator(), Br);
      Call->eraseFromParent();
      break;
    }
  }

  // Make sure `newpc` is still the first instruction when we have a jump target
  // (if not, create a new dedicated basic block); that otherwise would break
  // further assumptions when using `getBasicBlockPC` for model population.
  BasicBlock *Split = nullptr;
  Instruction *SplitPoint = nullptr;
  for (auto &BB : *OutlinedFunction.F) {
    if (Split == &BB)
      continue;

    Split = nullptr;
    for (auto &I : BB) {
      if (CallInst *Call = getCallTo(&I, "newpc")) {
        Value *IsJT = Call->getArgOperand(2);
        if (BB.getFirstNonPHI() != Call && getLimitedValue(IsJT) == 1) {

          if (isCallTo(Call->getPrevNode(), "function_call"))
            SplitPoint = Call->getPrevNode();
          else
            SplitPoint = Call;

          Split = BB.splitBasicBlock(SplitPoint, BB.getName() + Twine("_jt"));
          break;
        }
      }
    }
  }

  return OutlinedFunction;
}

bool StackAnalysis::runOnModule(Module &M) {
  Function &F = *M.getFunction("root");

  revng_log(PassesLog, "Starting StackAnalysis");

  auto &GCBI = getAnalysis<GeneratedCodeBasicInfoWrapperPass>().getGCBI();

  auto &LMP = getAnalysis<LoadModelWrapperPass>().get();

  // The stack analysis works function-wise. We consider two sets of functions:
  // first (Force == true) those that are highly likely to be real functions
  // (i.e., they have a direct call) and then (Force == false) all the remaining
  // candidates whose entry point is not included in any function of the first
  // set.

  std::vector<CFEP> Functions;
  model::Binary &Binary = *LMP.getWriteableModel();

  // Register all the static symbols
  for (const auto &F : Binary.Functions)
    Functions.emplace_back(GCBI.getBlockAt(F.Entry), true);

  // Register all the other candidate entry points
  for (BasicBlock &BB : F) {
    if (GCBI.getType(&BB) != BlockType::JumpTargetBlock)
      continue;

    auto It = llvm::find_if(Functions,
                            [&BB](const auto &E) { return E.Entry == &BB; });
    if (It != Functions.end())
      continue;

    uint32_t Reasons = GCBI.getJTReasons(&BB);
    bool IsCallee = hasReason(Reasons, JTReason::Callee);
    bool IsUnusedGlobalData = hasReason(Reasons, JTReason::UnusedGlobalData);
    bool IsMemoryStore = hasReason(Reasons, JTReason::MemoryStore);
    bool IsPCStore = hasReason(Reasons, JTReason::PCStore);
    bool IsReturnAddress = hasReason(Reasons, JTReason::ReturnAddress);
    bool IsLoadAddress = hasReason(Reasons, JTReason::LoadAddress);

    if (IsCallee) {
      // Called addresses are a strong hint
      Functions.emplace_back(&BB, true);
    } else if (not IsLoadAddress
               and (IsUnusedGlobalData
                    || (IsMemoryStore and not IsPCStore
                        and not IsReturnAddress))) {
      // TODO: keep IsReturnAddress?
      // Consider addresses found in global data that have not been used or
      // addresses that are not return addresses and do not end up in the PC
      // directly.
      Functions.emplace_back(&BB, false);
    }
  }

  for (CFEP &Function : Functions) {
    revng_log(CFEPLog,
              getName(Function.Entry) << (Function.Force ? " (forced)" : ""));
  }

  using BasicBlockToNodeMapTy = llvm::DenseMap<BasicBlock *, BasicBlockNode *>;
  BasicBlockToNodeMapTy BasicBlockNodeMap;

  // Queue to be populated with the CFEP
  llvm::SmallVector<BasicBlock *, 8> Worklist;
  SmallCallGraph CG;

  // Create an over-approximated call graph
  for (CFEP &Function : Functions) {
    BasicBlockNode Node{ Function.Entry };
    BasicBlockNode *GraphNode = CG.addNode(Node);
    BasicBlockNodeMap[Function.Entry] = GraphNode;
  }

  for (CFEP &Function : Functions) {
    llvm::SmallSet<BasicBlock *, 8> Visited;
    BasicBlockNode *StartNode = BasicBlockNodeMap[Function.Entry];
    revng_assert(StartNode != nullptr);
    Worklist.emplace_back(Function.Entry);

    while (!Worklist.empty()) {
      BasicBlock *Current = Worklist.pop_back_val();
      Visited.insert(Current);

      if (isFunctionCall(Current)) {
        // If not an indirect call, add the node to the CG
        if (BasicBlock *Callee = getFunctionCallCallee(Current)) {
          auto *Node = BasicBlockNodeMap[Callee];
          StartNode->addSuccessor(Node);
        }
        BasicBlock *Next = getFallthrough(Current);
        if (!Visited.count(Next))
          Worklist.push_back(Next);
      }

      for (BasicBlock *Successor : successors(Current)) {
        if (!GCBI::isPartOfRootDispatcher(Successor)
            && !Visited.count(Successor))
          Worklist.push_back(Successor);
      }
    }
  }

  // Create a root entry node for the call-graph, connect all the nodes to it,
  // and perform a post-order traversal. Keep in mind that adding a root node as
  // a predecessor to all nodes does not affect POT of any node, except the root
  // node itself.
  BasicBlockNode FakeNode{ nullptr };
  BasicBlockNode *RootNode = CG.addNode(FakeNode);
  CG.setEntryNode(RootNode);

  for (const auto &[_, Node] : BasicBlockNodeMap)
    RootNode->addSuccessor(Node);

  UniquedQueue<BasicBlockNode *> CFEPQueue;
  for (auto *Node : llvm::post_order(&CG)) {
    if (Node != RootNode)
      CFEPQueue.insert(Node);
  }

  // Dump the call-graph, if requested
  std::unique_ptr<raw_fd_ostream> OutputCG;
  if (CallGraphOutputPath.getNumOccurrences() == 1) {
    std::ifstream File(CallGraphOutputPath.c_str());
    if (File.is_open()) {
      int Status = std::remove(CallGraphOutputPath.c_str());
      revng_assert(Status == 0);
    }

    std::error_code EC;
    OutputCG = std::make_unique<raw_fd_ostream>(CallGraphOutputPath,
                                                EC,
                                                llvm::sys::fs::OF_Append);
    revng_assert(!EC);
    llvm::WriteGraph(*OutputCG, &CG);
  }

  // Collect all the ABI registers, leave out the stack pointer for the moment.
  // We will include it back later when refining ABI results.
  std::vector<llvm::GlobalVariable *> ABIRegisters;
  for (GlobalVariable *CSV : GCBI.abiRegisters())
    if (CSV != nullptr && !(GCBI.isSPReg(CSV)))
      ABIRegisters.emplace_back(CSV);

  // Default-constructed cache summary for indirect calls
  FunctionSummary DefaultSummary(model::FunctionType::Values::Regular,
                                 { ABIRegisters.begin(), ABIRegisters.end() },
                                 ABIAnalyses::ABIAnalysesResults(),
                                 {},
                                 GCBI.minimalFSO(),
                                 nullptr);
  FunctionAnalysisResults Properties(std::move(DefaultSummary));

  // Instantiate a CFEPAnalyzer object
  using CFEPA = CFEPAnalyzer<FunctionAnalysisResults>;
  CFEPA Analyzer(M, &GCBI, Properties, ABIRegisters);

  // Interprocedural analysis over the collected functions in post-order
  // traversal (leafs first).
  while (!CFEPQueue.empty()) {
    BasicBlockNode *EntryNode = CFEPQueue.pop();
    revng_log(StackAnalysisLog,
              "Analyzing Entry: " << EntryNode->BB->getName());

    // Intraprocedural analysis
    FunctionSummary AnalysisResult = Analyzer.analyze(EntryNode->BB);
    bool Changed = Properties.registerFunction(getBasicBlockPC(EntryNode->BB),
                                               std::move(AnalysisResult));

    // If we got improved results for a function, we need to recompute its
    // callers, and if a caller turns out to be fake, the callers of the fake
    // function too.
    if (Changed) {
      UniquedQueue<BasicBlockNode *> FakeFunctionWorklist;
      FakeFunctionWorklist.insert(EntryNode);

      while (!FakeFunctionWorklist.empty()) {
        BasicBlockNode *Node = FakeFunctionWorklist.pop();
        for (auto *Caller : Node->predecessors()) {
          if (Caller == RootNode)
            break;

          if (!Properties.isFakeFunction(getBasicBlockPC(Caller->BB)))
            CFEPQueue.insert(Caller);
          else
            FakeFunctionWorklist.insert(Caller);
        }
      }
    }
  }

  // Still OK?
  if (VerifyLog.isEnabled())
    revng_assert(llvm::verifyModule(M, &llvm::dbgs()) == false);

  // Propagate results between call-sites and functions
  interproceduralPropagation(Functions, Properties);

  // Initialize the cache where all the results will be accumulated
  Cache TheCache(&F, &GCBI);

  // Pool where the final results will be collected
  ResultsPool Results;

  // First analyze all the `Force`d functions (i.e., with an explicit direct
  // call)
  for (CFEP &Function : Functions) {
    if (Function.Force) {
      auto &GCBI = getAnalysis<GeneratedCodeBasicInfoWrapperPass>().getGCBI();
      InterproceduralAnalysis SA(TheCache, GCBI);
      SA.run(Function.Entry, Results);
    }
  }

  // Now analyze all the remaining candidates which are not already part of
  // another function
  std::set<BasicBlock *> Visited = Results.visitedBlocks();
  for (CFEP &Function : Functions) {
    if (not Function.Force and Visited.count(Function.Entry) == 0) {
      auto &GCBI = getAnalysis<GeneratedCodeBasicInfoWrapperPass>().getGCBI();
      InterproceduralAnalysis SA(TheCache, GCBI);
      SA.run(Function.Entry, Results);
    }
  }

  for (CFEP &Function : Functions) {
    using IFS = IntraproceduralFunctionSummary;
    BasicBlock *Entry = Function.Entry;
    llvm::Optional<const IFS *> Cached = TheCache.get(Entry);
    revng_assert(Cached or TheCache.isFakeFunction(Entry));

    // Has this function been analyzed already? If so, only now we register it
    // in the ResultsPool.
    FunctionType::Values Type;
    if (TheCache.isFakeFunction(Entry))
      Type = FunctionType::Fake;
    else if (TheCache.isNoReturnFunction(Entry))
      Type = FunctionType::NoReturn;
    else
      Type = FunctionType::Regular;

    // Regular functions need to be composed by at least a basic block
    if (Cached) {
      const IFS *Summary = *Cached;
      if (Type == FunctionType::Regular)
        revng_assert(Summary->BranchesType.size() != 0);

      Results.registerFunction(Entry, Type, Summary);
    } else {
      Results.registerFunction(Entry, Type, nullptr);
    }
  }

  GrandResult = Results.finalize(&M, &TheCache);

  if (ClobberedLog.isEnabled()) {
    for (auto &P : GrandResult.Functions) {
      ClobberedLog << getName(P.first) << ":";
      for (const llvm::GlobalVariable *CSV : P.second.ClobberedRegisters)
        ClobberedLog << " " << CSV->getName().data();
      ClobberedLog << DoLog;
    }
  }

  if (StackAnalysisLog.isEnabled()) {
    std::stringstream Output;
    GrandResult.dump(&M, Output);
    TextRepresentation = Output.str();
    revng_log(StackAnalysisLog, TextRepresentation);
  }

  revng_log(PassesLog, "Ending StackAnalysis");

  if (ABIAnalysisOutputPath.getNumOccurrences() == 1) {
    std::ofstream Output;
    serialize(pathToStream(ABIAnalysisOutputPath, Output));
  }

  commitToModel(GCBI, &F, GrandResult, Binary);

  return false;
}

void StackAnalysis::serializeMetadata(Function &F,
                                      GeneratedCodeBasicInfo &GCBI) {
  using namespace llvm;

  const FunctionsSummary &Summary = GrandResult;

  LLVMContext &Context = getContext(&F);
  QuickMetadata QMD(Context);

  // Temporary data structure so we can set all the `revng.func.member.of` in a
  // single shot at the end
  std::map<Instruction *, std::vector<Metadata *>> MemberOf;

  // Loop over all the detected functions
  for (const auto &P : Summary.Functions) {
    BasicBlock *Entry = P.first;
    const FunctionsSummary::FunctionDescription &Function = P.second;

    if (Entry == nullptr or Function.BasicBlocks.size() == 0)
      continue;

    MetaAddress EntryPC = getBasicBlockPC(Entry);

    //
    // Add `revng.func.entry`:
    // {
    //   name,
    //   address,
    //   type,
    //   { clobbered csv, ... },
    //   { { csv, argument, return value }, ... }
    // }
    //
    auto *TypeMD = QMD.get(FunctionType::getName(Function.Type));

    // Clobbered registers metadata
    std::vector<Metadata *> ClobberedMDs;
    for (GlobalVariable *ClobberedCSV : Function.ClobberedRegisters) {
      if (not GCBI.isServiceRegister(ClobberedCSV))
        ClobberedMDs.push_back(QMD.get(ClobberedCSV));
    }

    // Register slots metadata
    std::vector<Metadata *> SlotMDs;
    for (auto &P : Function.RegisterSlots) {
      if (GCBI.isServiceRegister(P.first))
        continue;

      auto *CSV = QMD.get(P.first);
      auto *Argument = QMD.get(P.second.Argument.valueName());
      auto *ReturnValue = QMD.get(P.second.ReturnValue.valueName());
      SlotMDs.push_back(QMD.tuple({ CSV, Argument, ReturnValue }));
    }

    // Create revng.func.entry metadata
    MDTuple *FunctionMD = QMD.tuple({ QMD.get(getName(Entry)),
                                      QMD.get(GCBI.toConstant(EntryPC)),
                                      TypeMD,
                                      QMD.tuple(ClobberedMDs),
                                      QMD.tuple(SlotMDs) });
    Entry->getTerminator()->setMetadata("revng.func.entry", FunctionMD);

    //
    // Create func.call
    //
    for (const FunctionsSummary::CallSiteDescription &CallSite :
         Function.CallSites) {
      Instruction *Call = CallSite.Call;

      // Register slots metadata
      std::vector<Metadata *> SlotMDs;
      for (auto &P : CallSite.RegisterSlots) {
        if (GCBI.isServiceRegister(P.first))
          continue;

        auto *CSV = QMD.get(P.first);
        auto *Argument = QMD.get(P.second.Argument.valueName());
        auto *ReturnValue = QMD.get(P.second.ReturnValue.valueName());
        SlotMDs.push_back(QMD.tuple({ CSV, Argument, ReturnValue }));
      }

      Call->setMetadata("func.call", QMD.tuple(QMD.tuple(SlotMDs)));
    }

    //
    // Create revng.func.member.of
    //

    // Loop over all the basic blocks composing the function
    for (const auto &P : Function.BasicBlocks) {
      BasicBlock *BB = P.first;
      BranchType::Values Type = P.second;

      auto *Pair = QMD.tuple({ FunctionMD, QMD.get(getName(Type)) });

      // Register that this block is associated to this function
      MemberOf[BB->getTerminator()].push_back(Pair);
    }
  }

  // Apply `revng.func.member.of`
  for (auto &P : MemberOf)
    P.first->setMetadata("revng.func.member.of", QMD.tuple(P.second));
}

} // namespace StackAnalysis
