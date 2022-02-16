/// \file EarlyFunctionAnalysis.cpp

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
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Value.h"
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

#include "revng/ABI/FunctionType.h"
#include "revng/ADT/KeyedObjectTraits.h"
#include "revng/ADT/Queue.h"
#include "revng/ADT/SortedVector.h"
#include "revng/ADT/ZipMapIterator.h"
#include "revng/BasicAnalyses/RemoveHelperCalls.h"
#include "revng/BasicAnalyses/RemoveNewPCCalls.h"
#include "revng/EarlyFunctionAnalysis/EarlyFunctionAnalysis.h"
#include "revng/EarlyFunctionAnalysis/FunctionMetadata.h"
#include "revng/Model/Architecture.h"
#include "revng/Model/Binary.h"
#include "revng/Model/NamedTypedRegister.h"
#include "revng/Model/Register.h"
#include "revng/Model/TypedRegister.h"
#include "revng/Support/Assert.h"
#include "revng/Support/CommandLine.h"
#include "revng/Support/IRHelpers.h"
#include "revng/Support/MetaAddress.h"

#include "ABIAnalyses/ABIAnalysis.h"

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

using FunctionEdgeTypeValue = efa::FunctionEdgeType::Values;
using FunctionTypeValue = model::FunctionType::Values;
using GCBI = GeneratedCodeBasicInfo;

using namespace llvm::cl;

static Logger<> EarlyFunctionAnalysisLog("earlyfunctionanalysis");

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

namespace EarlyFunctionAnalysis {

template<>
char EarlyFunctionAnalysis<true>::ID = 0;

template<>
char EarlyFunctionAnalysis<false>::ID = 0;

using ABIDetectionPass = RegisterPass<EarlyFunctionAnalysis<true>>;
static ABIDetectionPass X("detect-abi", "ABI Detection Pass", true, false);

using CFGCollectionPass = RegisterPass<EarlyFunctionAnalysis<false>>;
static CFGCollectionPass Y("collect-cfg", "CFG Collection Pass", true, true);

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
  SortedVector<efa::BasicBlock> CFG;
  std::optional<int64_t> ElectedFSO;
  llvm::Function *FakeFunction;

public:
  FunctionSummary(model::FunctionType::Values Type,
                  std::set<llvm::GlobalVariable *> ClobberedRegisters,
                  ABIAnalyses::ABIAnalysesResults ABIResults,
                  SortedVector<efa::BasicBlock> CFG,
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
  /// A default summary for indirect function calls is maintained. It has
  /// `Regular` type, it clobbers all the registers and it has default FSO.
  FunctionSummary DefaultSummary;

public:
  FunctionAnalysisResults(FunctionSummary DefaultSummary) :
    DefaultSummary(std::move(DefaultSummary)) {}

  FunctionSummary &insert(const MetaAddress &PC, FunctionSummary &&Summary) {
    auto It = FunctionsBucket.find(PC);
    revng_assert(It == FunctionsBucket.end());
    return FunctionsBucket.emplace(PC, std::move(Summary)).first->second;
  }

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

using BasicBlockQueue = UniquedQueue<BasicBlockNode *>;

/// An intraprocedural analysis storage.
///
/// Implementation of the intraprocedural stack analysis. It holds the
/// necessary information to detect the boundaries of a function, track
/// how the stack evolves within those functions, and detect the
/// callee-saved registers.
class FunctionEntrypointAnalyzer {
private:
  llvm::Module &M;
  llvm::LLVMContext &Context;
  GeneratedCodeBasicInfo *GCBI;
  ArrayRef<GlobalVariable *> ABICSVs;
  BasicBlockQueue *EntrypointsQueue;
  FunctionAnalysisResults &Oracle;
  TupleTree<model::Binary> &Binary;
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
  FunctionEntrypointAnalyzer(llvm::Module &,
                             GeneratedCodeBasicInfo *GCBI,
                             ArrayRef<GlobalVariable *>,
                             BasicBlockQueue *,
                             FunctionAnalysisResults &,
                             TupleTree<model::Binary> &);

public:
  void importModel();
  void runInterproceduralAnalysis();
  void interproceduralPropagation();
  void finalizeModel();
  void recoverCFG();
  void serializeFunctionMetadata();

private:
  /// The `analyze` method is the entry point of the intraprocedural analysis,
  /// and it is called on each function entrypoint until a fixed point is
  /// reached. It is responsible for performing the whole computation.
  FunctionSummary analyze(llvm::BasicBlock *BB, bool ShouldAnalyzeABI);

private:
  OutlinedFunction outlineFunction(llvm::BasicBlock *BB);
  void integrateFunctionCallee(llvm::BasicBlock *BB, MetaAddress);
  SortedVector<efa::BasicBlock> collectDirectCFG(OutlinedFunction *F);
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
                           SortedVector<efa::BasicBlock> &,
                           ABIAnalyses::ABIAnalysesResults &,
                           const std::set<llvm::GlobalVariable *> &,
                           bool ShouldAnalyzeABI);
  llvm::Function *createFakeFunction(llvm::BasicBlock *BB);
  UpcastablePointer<model::Type>
  buildPrototype(const FunctionSummary &, const efa::BasicBlock &);
  FunctionSummary importPrototype(model::FunctionType::Values, model::TypePath);

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
using FEA = FunctionEntrypointAnalyzer;

FEA::FunctionEntrypointAnalyzer(llvm::Module &M,
                                GeneratedCodeBasicInfo *GCBI,
                                ArrayRef<GlobalVariable *> ABICSVs,
                                BasicBlockQueue *EntrypointsQueue,
                                FunctionAnalysisResults &Oracle,
                                TupleTree<model::Binary> &Binary) :
  M(M),
  Context(M.getContext()),
  GCBI(GCBI),
  ABICSVs(ABICSVs),
  EntrypointsQueue(EntrypointsQueue),
  Oracle(Oracle),
  Binary(Binary),
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
    for (const auto &Reg : ABICSVs)
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

void FunctionEntrypointAnalyzer::serializeFunctionMetadata() {
  using namespace llvm;

  for (const auto &Function : Binary->Functions) {
    if (Function.Type == FunctionTypeValue::Invalid
        || Function.Type == FunctionTypeValue::Fake)
      continue;

    auto &CFG = Oracle.at(Function.Entry).CFG;
    BasicBlock *BB = GCBI->getBlockAt(Function.Entry);
    std::string Buffer;
    {
      efa::FunctionMetadata FM(Function.Entry);
      raw_string_ostream Stream(Buffer);
      for (efa::BasicBlock Edge : CFG)
        FM.ControlFlowGraph.insert(Edge);

      FM.verify(*Binary, true);
      serialize(Stream, FM);
    }

    Instruction *Term = BB->getTerminator();
    MDNode *Node = MDNode::get(Context, MDString::get(Context, Buffer));
    Term->setMetadata(FunctionMetadataMDName, Node);
  }
}

FunctionSummary
FunctionEntrypointAnalyzer::importPrototype(model::FunctionType::Values Type,
                                            model::TypePath Prototype) {
  using namespace llvm;
  using namespace model;
  using Register = model::Register::Values;
  using State = abi::RegisterState::Values;

  FunctionSummary Summary(Type,
                          { ABICSVs.begin(), ABICSVs.end() },
                          ABIAnalyses::ABIAnalysesResults(),
                          {},
                          0,
                          nullptr);

  for (GlobalVariable *CSV : ABICSVs) {
    Summary.ABIResults.ArgumentsRegisters[CSV] = State::No;
    Summary.ABIResults.FinalReturnValuesRegisters[CSV] = State::No;
  }

  auto Layout = abi::FunctionType::Layout::make(Prototype);

  for (const auto &ArgumentLayout : Layout.Arguments) {
    for (Register ArgumentRegister : ArgumentLayout.Registers) {
      StringRef Name = model::Register::getCSVName(ArgumentRegister);
      if (GlobalVariable *CSV = M.getGlobalVariable(Name, true))
        Summary.ABIResults.ArgumentsRegisters.at(CSV) = State::Yes;
    }
  }

  for (Register ReturnValueRegister : Layout.ReturnValue.Registers) {
    StringRef Name = model::Register::getCSVName(ReturnValueRegister);
    if (GlobalVariable *CSV = M.getGlobalVariable(Name, true))
      Summary.ABIResults.FinalReturnValuesRegisters.at(CSV) = State::Yes;
  }

  std::set<llvm::GlobalVariable *> PreservedRegisters;
  for (Register CalleeSavedRegister : Layout.CalleeSavedRegisters) {
    StringRef Name = model::Register::getCSVName(CalleeSavedRegister);
    if (GlobalVariable *CSV = M.getGlobalVariable(Name, true))
      PreservedRegisters.insert(CSV);
  }

  std::erase_if(Summary.ClobberedRegisters, [&](const auto &E) {
    return PreservedRegisters.count(E) != 0;
  });

  Summary.ElectedFSO = Layout.FinalStackOffset;
  return Summary;
}

void FunctionEntrypointAnalyzer::importModel() {
  // Import existing functions from model
  for (const model::Function &Function : Binary->Functions) {
    if (Function.Type == model::FunctionType::Invalid)
      continue;

    Oracle.insert(Function.Entry,
                  importPrototype(Function.Type, Function.Prototype));
  }

  // Re-create fake functions, should they exist
  for (const model::Function &Function : Binary->Functions) {
    if (Function.Type != FunctionTypeValue::Fake)
      continue;

    auto &Summary = Oracle.at(Function.Entry);
    revng_assert(Summary.Type == FunctionTypeValue::Fake);
    Summary.FakeFunction = createFakeFunction(GCBI->getBlockAt(Function.Entry));
  }
}

UpcastablePointer<model::Type>
FunctionEntrypointAnalyzer::buildPrototype(const FunctionSummary &Summary,
                                           const efa::BasicBlock &Block) {
  using namespace model;
  using RegisterState = abi::RegisterState::Values;

  auto NewType = makeType<RawFunctionType>();
  auto &CallType = *llvm::cast<RawFunctionType>(NewType.get());
  {
    auto ArgumentsInserter = CallType.Arguments.batch_insert();
    auto ReturnValuesInserter = CallType.ReturnValues.batch_insert();

    bool Found = false;
    for (const auto &[PC, CallSites] : Summary.ABIResults.CallSites) {
      if (PC != Block.Start)
        continue;

      revng_assert(!Found);
      Found = true;

      for (const auto &[Arg, RV] :
           zipmap_range(CallSites.ArgumentsRegisters,
                        CallSites.ReturnValuesRegisters)) {
        auto *CSV = Arg == nullptr ? RV->first : Arg->first;
        RegisterState RSArg = Arg == nullptr ? RegisterState::Maybe :
                                               Arg->second;
        RegisterState RSRV = RV == nullptr ? RegisterState::Maybe : RV->second;

        auto RegisterID = model::Register::fromCSVName(CSV->getName(),
                                                       Binary->Architecture);
        if (RegisterID == Register::Invalid || CSV == GCBI->spReg())
          continue;

        auto *CSVType = CSV->getType()->getPointerElementType();
        auto CSVSize = CSVType->getIntegerBitWidth() / 8;
        if (abi::RegisterState::shouldEmit(RSArg)) {
          NamedTypedRegister TR(RegisterID);
          TR.Type = {
            Binary->getPrimitiveType(PrimitiveTypeKind::Generic, CSVSize), {}
          };
          ArgumentsInserter.insert(TR);
        }

        if (abi::RegisterState::shouldEmit(RSRV)) {
          TypedRegister TR(RegisterID);
          TR.Type = {
            Binary->getPrimitiveType(PrimitiveTypeKind::Generic, CSVSize), {}
          };
          ReturnValuesInserter.insert(TR);
        }
      }
    }
    revng_assert(Found);

    CallType.PreservedRegisters = {};
    CallType.FinalStackOffset = 0;
  }

  return NewType;
}

/// Finish the population of the model by building the prototype
void FunctionEntrypointAnalyzer::finalizeModel() {
  using namespace model;
  using RegisterState = abi::RegisterState::Values;

  // Fill up the model and build its prototype for each function
  std::set<model::Function *> Functions;
  for (model::Function &Function : Binary->Functions) {
    if (Function.Type != model::FunctionType::Invalid)
      continue;

    MetaAddress EntryPC = Function.Entry;
    revng_assert(EntryPC.isValid());
    auto &Summary = Oracle.at(EntryPC);
    Function.Type = Summary.Type;

    auto NewType = makeType<RawFunctionType>();
    auto &FunctionType = *llvm::cast<RawFunctionType>(NewType.get());
    {
      auto ArgumentsInserter = FunctionType.Arguments.batch_insert();
      auto ReturnValuesInserter = FunctionType.ReturnValues.batch_insert();
      auto PreservedRegistersInserter = FunctionType.PreservedRegisters
                                          .batch_insert();

      // Argument and return values
      for (const auto &[Arg, RV] :
           zipmap_range(Summary.ABIResults.ArgumentsRegisters,
                        Summary.ABIResults.FinalReturnValuesRegisters)) {
        auto *CSV = Arg == nullptr ? RV->first : Arg->first;
        RegisterState RSArg = Arg == nullptr ? RegisterState::Maybe :
                                               Arg->second;
        RegisterState RSRV = RV == nullptr ? RegisterState::Maybe : RV->second;

        auto RegisterID = model::Register::fromCSVName(CSV->getName(),
                                                       Binary->Architecture);
        if (RegisterID == Register::Invalid || CSV == GCBI->spReg())
          continue;

        auto *CSVType = CSV->getType()->getPointerElementType();
        auto CSVSize = CSVType->getIntegerBitWidth() / 8;

        if (abi::RegisterState::shouldEmit(RSArg)) {
          NamedTypedRegister TR(RegisterID);
          TR.Type = {
            Binary->getPrimitiveType(PrimitiveTypeKind::Generic, CSVSize), {}
          };
          ArgumentsInserter.insert(TR);
        }

        if (abi::RegisterState::shouldEmit(RSRV)) {
          TypedRegister TR(RegisterID);
          TR.Type = {
            Binary->getPrimitiveType(PrimitiveTypeKind::Generic, CSVSize), {}
          };
          ReturnValuesInserter.insert(TR);
        }
      }

      // Preserved registers
      std::set<llvm::GlobalVariable *> PreservedRegisters(ABICSVs.begin(),
                                                          ABICSVs.end());
      std::erase_if(PreservedRegisters, [&](const auto &E) {
        auto End = Summary.ClobberedRegisters.end();
        return Summary.ClobberedRegisters.find(E) != End;
      });

      for (auto *CSV : PreservedRegisters) {
        auto RegisterID = model::Register::fromCSVName(CSV->getName(),
                                                       Binary->Architecture);
        if (RegisterID == Register::Invalid)
          continue;

        PreservedRegistersInserter.insert(RegisterID);
      }

      // Final stack offset
      FunctionType.FinalStackOffset = Summary.ElectedFSO.has_value() ?
                                        *Summary.ElectedFSO :
                                        0;
    }

    Function.Prototype = Binary->recordNewType(std::move(NewType));
    Functions.insert(&Function);
  }

  // Finish up the CFG
  for (auto &Function : Functions) {
    if (Function->Type == FunctionTypeValue::Fake)
      continue;

    auto &Summary = Oracle.at(Function->Entry);
    for (auto &Block : Summary.CFG) {
      for (auto &Edge : Block.Successors) {
        llvm::StringRef SymbolName;

        if (Edge->Destination.isValid()) {
          auto *Successor = GCBI->getBlockAt(Edge->Destination);
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

        if (efa::FunctionEdgeType::isCall(Edge->Type)) {
          auto *CE = llvm::cast<efa::CallEdge>(Edge.get());
          const auto IDF = Binary->ImportedDynamicFunctions;
          bool IsDynamicCall = (not SymbolName.empty()
                                and IDF.count(SymbolName.str()) != 0);

          if (IsDynamicCall) {
            // It's a dynamic function call
            revng_assert(CE->Type == efa::FunctionEdgeType::FunctionCall);
            CE->Destination = MetaAddress::invalid();
            CE->DynamicFunction = SymbolName.str();

            // The prototype must not exist among the ones of the call sites of
            // the function, it is implicitly the one of the callee.
            revng_assert(not Function->CallSitePrototypes.count(Block.Start));
          } else if (CE->Destination.isValid()) {
            // It's a simple direct function call
            revng_assert(CE->Type == efa::FunctionEdgeType::FunctionCall);

            // The prototype must not exist among the ones of the call sites of
            // the function, it is implicitly the one of the callee.
            revng_assert(not Function->CallSitePrototypes.count(Block.Start));
          } else {
            // It's an indirect call: forge a new prototype
            auto Prototype = buildPrototype(Summary, Block);
            auto TypedPrototype = Binary->recordNewType(std::move(Prototype));
            auto PrototypeInserter = Function->CallSitePrototypes
                                       .batch_insert();
            PrototypeInserter.insert({ Block.Start, TypedPrototype });
          }
        }
      }
    }

    efa::FunctionMetadata FM(Function->Entry, Summary.CFG);
    FM.verify(*Binary, true);
  }

  revng_check(Binary->verify(true));
}

static void combineCrossCallSites(auto &CallSite, auto &Callee) {
  using namespace ABIAnalyses;
  using RegisterState = abi::RegisterState::Values;

  for (auto &[FuncArg, CSArg] :
       zipmap_range(Callee.ArgumentsRegisters, CallSite.ArgumentsRegisters)) {
    auto *CSV = FuncArg == nullptr ? CSArg->first : FuncArg->first;
    auto RSFArg = FuncArg == nullptr ? RegisterState::Maybe : FuncArg->second;
    auto RSCSArg = CSArg == nullptr ? RegisterState::Maybe : CSArg->second;

    Callee.ArgumentsRegisters[CSV] = combine(RSFArg, RSCSArg);
  }
}

/// Perform cross-call site propagation
void FunctionEntrypointAnalyzer::interproceduralPropagation() {
  for (model::Function &Function : Binary->Functions) {
    auto &Summary = Oracle.at(Function.Entry);
    for (auto &[PC, CallSite] : Summary.ABIResults.CallSites) {
      if (PC == Function.Entry)
        combineCrossCallSites(CallSite, Summary.ABIResults);
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

static UpcastablePointer<efa::FunctionEdgeBase>
makeEdge(MetaAddress Destination, efa::FunctionEdgeType::Values Type) {
  efa::FunctionEdge *Result = nullptr;
  using ReturnType = UpcastablePointer<efa::FunctionEdgeBase>;

  if (efa::FunctionEdgeType::isCall(Type))
    return ReturnType::make<efa::CallEdge>(Destination, Type);
  else
    return ReturnType::make<efa::FunctionEdge>(Destination, Type);
};

static MetaAddress getFinalAddressOfBasicBlock(llvm::BasicBlock *BB) {
  auto [End, Size] = getPC(BB->getTerminator());
  return End + Size;
}

SortedVector<efa::BasicBlock>
FunctionEntrypointAnalyzer::collectDirectCFG(OutlinedFunction *F) {
  using namespace llvm;

  SortedVector<efa::BasicBlock> CFG;

  for (BasicBlock &BB : *F->F) {
    if (GCBI::isJumpTarget(&BB)) {
      MetaAddress Start = getBasicBlockPC(&BB);
      efa::BasicBlock Block{ Start };
      Block.End = getFinalAddressOfBasicBlock(&BB);

      OnceQueue<BasicBlock *> Queue;
      Queue.insert(&BB);

      // A JT with no successors?
      if (isa<UnreachableInst>(BB.getTerminator())) {
        auto Type = efa::FunctionEdgeType::Unreachable;
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
                                 efa::FunctionEdgeType::DirectBranch);
            Block.Successors.insert(Edge);
          } else if (F->UnexpectedPCCloned == Succ && succ_size(Current) == 1) {
            // Need to create an edge only when `unexpectedpc` is the unique
            // successor of the current basic block.
            auto Edge = makeEdge(MetaAddress::invalid(),
                                 efa::FunctionEdgeType::LongJmp);
            Block.Successors.insert(Edge);
          } else {
            Instruction *I = &(*Succ->begin());

            if (isa<ReturnInst>(I)) {
              // Did we meet the end of the cloned function? Do nothing
              revng_assert(Succ->getInstList().size() == 1);
            } else if (auto *Call = getCallTo(I, PreHookMarker.F)) {
              MetaAddress Destination;
              efa::FunctionEdgeType::Values Type;
              auto *CalleePC = Call->getArgOperand(1);

              // Direct or indirect call?
              if (isa<ConstantStruct>(CalleePC)) {
                auto AddressPC = MetaAddress::fromConstant(CalleePC);
                // Does the function exist within the model?
                auto It = Binary->Functions.find(AddressPC);
                if (It != Binary->Functions.end()) {
                  Destination = AddressPC;
                  Type = efa::FunctionEdgeType::FunctionCall;
                } else {
                  Destination = MetaAddress::invalid();
                  Type = efa::FunctionEdgeType::IndirectCall;
                }
              } else {
                Destination = MetaAddress::invalid();
                Type = efa::FunctionEdgeType::IndirectCall;
              }

              auto Edge = makeEdge(Destination, Type);
              auto DestTy = Oracle.getFunctionType(Destination);
              if (DestTy == FunctionTypeValue::NoReturn) {
                auto *CE = cast<efa::CallEdge>(Edge.get());
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
                                   efa::FunctionEdgeType::FakeFunctionCall);
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

void FEA::initMarkersForABI(OutlinedFunction *OutlinedFunction,
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

std::set<llvm::GlobalVariable *>
FunctionEntrypointAnalyzer::findWrittenRegisters(llvm::Function *F) {
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

void FEA::createIBIMarker(OutlinedFunction *OutlinedFunction,
                          SmallVectorImpl<Instruction *> &SV,
                          llvm::IRBuilder<> &IRB) {
  using namespace llvm;

  StructType *MetaAddressTy = MetaAddress::getStruct(&M);
  IRB.SetInsertPoint(&OutlinedFunction->F->getEntryBlock().front());
  auto *IntPtrTy = GCBI->spReg()->getType();
  auto *IntTy = GCBI->spReg()->getType()->getElementType();

  // At the entry of the function, load the initial value of stack pointer,
  // program counter and ABI registers used within this function.
  auto *SPI = IRB.CreateLoad(GCBI->spReg());
  auto *SPPtr = IRB.CreateIntToPtr(SPI, IntPtrTy);

  Value *RA = IRB.CreateLoad(GCBI->raReg() ? GCBI->raReg() : SPPtr);

  auto ToLLVMArchitecture = model::Architecture::toLLVMArchitecture;
  auto LLVMArchitecture = ToLLVMArchitecture(Binary->Architecture);
  std::array<Value *, 4> DissectedPC = PCH->dissectJumpablePC(IRB,
                                                              RA,
                                                              LLVMArchitecture);

  auto *PCI = MetaAddress::composeIntegerPC(IRB,
                                            DissectedPC[0],
                                            DissectedPC[1],
                                            DissectedPC[2],
                                            DissectedPC[3]);

  SmallVector<Value *, 16> CSVI;
  Type *IsRetTy = Type::getInt128Ty(Context);
  SmallVector<Type *, 16> ArgTypes = { IsRetTy, IntTy, MetaAddressTy };
  for (auto *CSR : ABICSVs) {
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
    for (auto *CSR : ABICSVs) {
      auto *V = IRB.CreateLoad(CSR);
      CSVE.emplace_back(V);
    }

    auto *SPE = IRB.CreateLoad(GCBI->spReg());

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

void FEA::opaqueBranchConditions(llvm::Function *F, llvm::IRBuilder<> &IRB) {
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

void FEA::materializePCValues(llvm::Function *F, llvm::IRBuilder<> &IRB) {
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

void FunctionEntrypointAnalyzer::runOptimizationPipeline(llvm::Function *F) {
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
    FAM.registerPass([this] {
      using LMA = LoadModelAnalysis;
      return LMA::fromModelWrapper(Binary);
    });
    FAM.registerPass([&] { return GeneratedCodeBasicInfoAnalysis(); });
    FAM.registerPass([&] { return ModuleAnalysisManagerFunctionProxy(MAM); });

    PassBuilder PB;
    PB.registerFunctionAnalyses(FAM);
    PB.registerModuleAnalyses(MAM);

    FPM.run(*F, FAM);
  }
}

llvm::Function *
FunctionEntrypointAnalyzer::createFakeFunction(llvm::BasicBlock *Entry) {
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

void FunctionEntrypointAnalyzer::recoverCFG() {
  for (const auto &Function : Binary->Functions) {
    // No CFG will be recovered for `Fake` or `Invalid` functions
    if (Function.Type == FunctionTypeValue::Invalid
        || Function.Type == FunctionTypeValue::Fake)
      continue;

    auto *Entry = GCBI->getBlockAt(Function.Entry);

    // Recover the control-flow graph of the function
    auto &Summary = Oracle.at(Function.Entry);
    Summary.CFG = std::move(analyze(Entry, false).CFG);
  }
}

FunctionSummary
FunctionEntrypointAnalyzer::analyze(BasicBlock *Entry, bool ShouldAnalyzeABI) {
  using namespace llvm;
  using namespace ABIAnalyses;

  IRBuilder<> Builder(M.getContext());
  ABIAnalysesResults ABIResults;
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

  if (ShouldAnalyzeABI) {
    // Run ABI-independent data-flow analyses
    ABIResults = analyzeOutlinedFunction(OutlinedFunction.F,
                                         *GCBI,
                                         PreHookMarker.F,
                                         PostHookMarker.F,
                                         RetHookMarker.F);
  }

  // Recompute the DomTree for the current outlined function due to split
  // basic blocks.
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
                               WrittenRegisters,
                               ShouldAnalyzeABI);

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
  using RegisterState = abi::RegisterState::Values;

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

FunctionSummary
FEA::milkInfo(OutlinedFunction *OutlinedFunction,
              SortedVector<efa::BasicBlock> &CFG,
              ABIAnalyses::ABIAnalysesResults &ABIResults,
              const std::set<GlobalVariable *> &WrittenRegisters,
              bool ShouldAnalyzeABI) {
  using namespace llvm;

  SmallVector<std::pair<CallBase *, int64_t>, 4> MaybeReturns;
  SmallVector<std::pair<CallBase *, int64_t>, 4> NotReturns;
  SmallVector<std::pair<CallBase *, FunctionEdgeTypeValue>, 4> IBIResult;
  std::set<GlobalVariable *> CalleeSavedRegs;
  std::set<GlobalVariable *> ClobberedRegs(ABICSVs.begin(), ABICSVs.end());

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
            CalleeSavedRegs.insert(ABICSVs[Idx - 3]);
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
    efa::BasicBlock &Block = CFG.at(PC);
    Block.Successors.insert(makeEdge(MetaAddress::invalid(), EdgeType));
  }

  // Empty CFG if function is fake
  if (Type == FunctionTypeValue::Fake)
    CFG.clear();

  if (ShouldAnalyzeABI) {
    // We say that a register is callee-saved when, besides being preserved by
    // the callee, there is at least a write onto this register.
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

    ABIResults.dump(EarlyFunctionAnalysisLog);
  }

  return FunctionSummary(Type,
                         std::move(ClobberedRegs),
                         std::move(ABIResults),
                         std::move(CFG),
                         WinFSO,
                         nullptr);
}

void FunctionEntrypointAnalyzer::integrateFunctionCallee(llvm::BasicBlock *BB,
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
    if (auto MaybeFSO = Oracle.getElectedFSO(Next); MaybeFSO) {
      GlobalVariable *SPCSV = GCBI->spReg();
      auto *StackPointer = Builder.CreateLoad(SPCSV);
      Value *Offset = ConstantInt::get(StackPointer->getPointerOperandType()
                                         ->getPointerElementType(),
                                       *MaybeFSO);
      auto *AdjustedStackPointer = Builder.CreateAdd(StackPointer, Offset);
      Builder.CreateStore(AdjustedStackPointer, SPCSV);
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
    revng_log(EarlyFunctionAnalysisLog,
              "Has callee " << FakeFunction->getName() << "been inlined? "
                            << Status);
    break;
  }

  default:
    revng_abort();
  }
}

OutlinedFunction
FunctionEntrypointAnalyzer::outlineFunction(llvm::BasicBlock *Entry) {
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
        if (not isPartOfRootDispatcher(Successor))
          Queue.insert(Successor);
      }
    }
  }

  // Create a copy of all the basic blocks to outline in `root`
  ValueToValueMapTy VMap;
  SmallVector<BasicBlock *, 8> BlocksToExtract;

  auto *AnyPCBB = GCBI->anyPC();
  auto *UnexpectedPCBB = GCBI->unexpectedPC();

  for (const auto &BB : BlocksToClone) {
    BasicBlock *Cloned = CloneBasicBlock(BB, VMap, Twine("_cloned"), Root);

    VMap[BB] = Cloned;
    BlocksToExtract.emplace_back(Cloned);
  }

  auto AnyPCIt = VMap.find(AnyPCBB);
  if (AnyPCIt != VMap.end())
    OutlinedFunction.AnyPCCloned = cast<BasicBlock>(AnyPCIt->second);

  auto UnexpPCIt = VMap.find(UnexpectedPCBB);
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

void FunctionEntrypointAnalyzer::runInterproceduralAnalysis() {
  while (!EntrypointsQueue->empty()) {
    BasicBlockNode *EntryNode = EntrypointsQueue->pop();
    revng_log(EarlyFunctionAnalysisLog,
              "Analyzing Entry: " << EntryNode->BB->getName());

    // Intraprocedural analysis
    FunctionSummary AnalysisResult = analyze(EntryNode->BB, true);
    bool Changed = Oracle.registerFunction(getBasicBlockPC(EntryNode->BB),
                                           std::move(AnalysisResult));

    // If we got improved results for a function, we need to recompute its
    // callers, and if a caller turns out to be fake, the callers of the fake
    // function too.
    if (Changed) {
      BasicBlockQueue FakeFunctionWorklist;
      FakeFunctionWorklist.insert(EntryNode);

      while (!FakeFunctionWorklist.empty()) {
        BasicBlockNode *Node = FakeFunctionWorklist.pop();
        for (auto *Caller : Node->predecessors()) {
          // Root node?
          if (Caller->BB == nullptr)
            break;

          if (!Oracle.isFakeFunction(getBasicBlockPC(Caller->BB)))
            EntrypointsQueue->insert(Caller);
          else
            FakeFunctionWorklist.insert(Caller);
        }
      }
    }
  }
}

template<bool ShouldAnalyzeABI>
bool EarlyFunctionAnalysis<ShouldAnalyzeABI>::runOnModule(Module &M) {
  revng_log(PassesLog, "Starting EarlyFunctionAnalysis");

  auto &GCBI = getAnalysis<GeneratedCodeBasicInfoWrapperPass>().getGCBI();
  auto &LMP = getAnalysis<LoadModelWrapperPass>().get();

  TupleTree<model::Binary> &Binary = LMP.getWriteableModel();

  using BasicBlockToNodeMap = llvm::DenseMap<BasicBlock *, BasicBlockNode *>;
  BasicBlockToNodeMap BasicBlockNodeMap;

  // Temporary worklist to collect the function entrypoints
  llvm::SmallVector<BasicBlock *, 8> Worklist;
  SmallCallGraph CG;

  // Create an over-approximated call graph
  for (const auto &Function : Binary->Functions) {
    auto *Entry = GCBI.getBlockAt(Function.Entry);
    BasicBlockNode Node{ Entry };
    BasicBlockNode *GraphNode = CG.addNode(Node);
    BasicBlockNodeMap[Entry] = GraphNode;
  }

  for (const auto &Function : Binary->Functions) {
    llvm::SmallSet<BasicBlock *, 8> Visited;
    auto *Entry = GCBI.getBlockAt(Function.Entry);
    BasicBlockNode *StartNode = BasicBlockNodeMap[Entry];
    revng_assert(StartNode != nullptr);
    Worklist.emplace_back(Entry);

    while (!Worklist.empty()) {
      BasicBlock *Current = Worklist.pop_back_val();
      Visited.insert(Current);

      if (isFunctionCall(Current)) {
        // If not an indirect call, add the node to the CG
        if (BasicBlock *Callee = getFunctionCallCallee(Current)) {
          BasicBlockNode *GraphNode = nullptr;
          auto It = BasicBlockNodeMap.find(Callee);
          if (It != BasicBlockNodeMap.end())
            StartNode->addSuccessor(It->second);
        }
        BasicBlock *Next = getFallthrough(Current);
        if (!Visited.count(Next))
          Worklist.push_back(Next);
      }

      for (BasicBlock *Successor : successors(Current)) {
        if (not isPartOfRootDispatcher(Successor) && !Visited.count(Successor))
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

  // Create an over-approximated call graph of the program. A queue of all the
  // function entrypoints is maintained.
  BasicBlockQueue EntrypointsQueue;
  for (auto *Node : llvm::post_order(&CG)) {
    if (Node == RootNode)
      continue;

    // The intraprocedural analysis will be scheduled only for those functions
    // which have `Invalid` as type.
    auto &Function = Binary->Functions.at(getBasicBlockPC(Node->BB));
    if (Function.Type == model::FunctionType::Invalid)
      EntrypointsQueue.insert(Node);
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
  std::vector<llvm::GlobalVariable *> ABICSVs;
  for (GlobalVariable *CSV : GCBI.abiRegisters())
    if (CSV != nullptr && !(GCBI.isSPReg(CSV)))
      ABICSVs.emplace_back(CSV);

  // Default-constructed cache summary for indirect calls
  unsigned MinimalFSO;
  {
    using namespace model::Architecture;
    MinimalFSO = getMinimalFinalStackOffset(Binary->Architecture);
  }

  FunctionSummary DefaultSummary(model::FunctionType::Values::Regular,
                                 { ABICSVs.begin(), ABICSVs.end() },
                                 ABIAnalyses::ABIAnalysesResults(),
                                 {},
                                 MinimalFSO,
                                 nullptr);

  using FAR = FunctionAnalysisResults;
  FAR Properties(std::move(DefaultSummary));

  // Instantiate a FunctionEntrypointAnalyzer object
  FEA Analyzer(M, &GCBI, ABICSVs, &EntrypointsQueue, Properties, Binary);

  // Prepopulate the cache with functions from model and recreate fake functions
  Analyzer.importModel();

  // EarlyFunctionAnalysis can be invoked with two option: via `--detect-abi` in
  // order to schedule a full ABI analysis for each function entry-point
  // detected, or via `--collect-cfg`, for control-flow graph recovery. In doing
  // so, the following property is obtained: `--detect-abi` writes the model but
  // not the IR, whereas `--collect-cfg` writes the IR but not the model.

  if (!ShouldAnalyzeABI) {
    // Recover the control-flow graph
    Analyzer.recoverCFG();

    // Serialize function metadata, CFG included, to IR
    Analyzer.serializeFunctionMetadata();
  } else {
    // Interprocedural analysis over the collected functions in post-order
    // traversal (leafs first).
    Analyzer.runInterproceduralAnalysis();

    // Propagate results between call-sites and functions
    Analyzer.interproceduralPropagation();

    // Commit the results onto the model
    Analyzer.finalizeModel();
  }

  // Still OK?
  if (VerifyLog.isEnabled())
    revng_assert(llvm::verifyModule(M, &llvm::dbgs()) == false);

  return false;
}

} // namespace EarlyFunctionAnalysis
