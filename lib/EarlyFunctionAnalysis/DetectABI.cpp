/// \file DetectABI.cpp

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <fstream>

#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/DOTGraphTraits.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/GraphWriter.h"

#include "revng/ABI/Definition.h"
#include "revng/ADT/Queue.h"
#include "revng/BasicAnalyses/GeneratedCodeBasicInfo.h"
#include "revng/EarlyFunctionAnalysis/ABIAnalysis.h"
#include "revng/EarlyFunctionAnalysis/CFGAnalyzer.h"
#include "revng/EarlyFunctionAnalysis/CallGraph.h"
#include "revng/EarlyFunctionAnalysis/CollectFunctionsFromCalleesPass.h"
#include "revng/EarlyFunctionAnalysis/CollectFunctionsFromUnusedAddressesPass.h"
#include "revng/EarlyFunctionAnalysis/Common.h"
#include "revng/EarlyFunctionAnalysis/DetectABI.h"
#include "revng/EarlyFunctionAnalysis/FunctionMetadata.h"
#include "revng/EarlyFunctionAnalysis/FunctionSummaryOracle.h"
#include "revng/MFP/MFP.h"
#include "revng/MFP/SetLattices.h"
#include "revng/Model/Binary.h"
#include "revng/Model/Generated/Early/Register.h"
#include "revng/Model/Register.h"
#include "revng/Pipeline/Pipe.h"
#include "revng/Pipeline/RegisterAnalysis.h"
#include "revng/Pipes/Kinds.h"
#include "revng/Pipes/LLVMAnalysisImplementation.h"

#include "ABISummary.h"
#include "InterproceduralAnalysis.h"
#include "InterproceduralGraph.h"

using namespace llvm;
using namespace llvm::cl;

static opt<std::string> CallGraphOutputPath("cg-output",
                                            desc("Dump to disk the recovered "
                                                 "call graph."),
                                            value_desc("filename"));

enum ABIEnforcementOption {
  NoABIEnforcement = 0,
  SoftABIEnforcement,
  FullABIEnforcement,
};

using ABIOpt = ABIEnforcementOption;
static opt<ABIOpt> ABIEnforcement("abi-enforcement-level",
                                  desc("ABI refinement preferences."),
                                  values(clEnumValN(NoABIEnforcement,
                                                    "no",
                                                    "Do not enforce "
                                                    "ABI-specific "
                                                    "information."),
                                         clEnumValN(SoftABIEnforcement,
                                                    "soft",
                                                    "Enforce ABI-specific "
                                                    "information, but allows "
                                                    "for incompatibility, if "
                                                    "unsure about the ABI."),
                                         clEnumValN(FullABIEnforcement,
                                                    "full",
                                                    "Enforce ABI-specific "
                                                    "information, and do not "
                                                    "allow for any possible "
                                                    "violations of the ABI "
                                                    "found.")),
                                  init(ABIOpt::FullABIEnforcement));

static Logger<> Log("detect-abi");

class DetectABIAnalysis {
private:
  template<typename... T>
  using Impl = revng::pipes::LLVMAnalysisImplementation<T...>;

  Impl<CollectFunctionsFromCalleesWrapperPass,
       efa::DetectABIPass,
       CollectFunctionsFromUnusedAddressesWrapperPass,
       efa::DetectABIPass>
    Implementation;

public:
  static constexpr auto Name = "DetectABI";

  std::vector<std::vector<pipeline::Kind *>> AcceptedKinds = {
    { &revng::kinds::Root }
  };

  void print(const pipeline::Context &Ctx,
             llvm::raw_ostream &OS,
             llvm::ArrayRef<std::string> ContainerNames) const {
    Implementation.print(Ctx, OS, ContainerNames);
  }

  void run(const pipeline::Context &Ctx, pipeline::LLVMContainer &Container) {
    Implementation.run(Ctx, Container);
  }
};

static pipeline::RegisterAnalysis<DetectABIAnalysis> A1;

namespace efa {

using BasicBlockQueue = UniquedQueue<const BasicBlockNode *>;

class DetectABI {
private:
  using CSVSet = std::set<llvm::GlobalVariable *>;
  using IA = InterproceduralAnalysis;

private:
  llvm::Module &M;
  llvm::LLVMContext &Context;
  GeneratedCodeBasicInfo &GCBI;
  TupleTree<model::Binary> &Binary;
  FunctionSummaryOracle &Oracle;
  CFGAnalyzer &Analyzer;

  BasicBlockQueue EntrypointsQueue;

  CallGraph ApproximateCallGraph;

  TemporaryOpaqueFunction EntryHook;

  std::map<MetaAddress, OutlinedFunction> OutlinedFunctions;

  std::map<Node::AddressType, ABISummary> FinalResults;

  std::map<MetaAddress, ABIAnalyses::ABIAnalysesResults> PartialResults;

public:
  DetectABI(llvm::Module &M,
            GeneratedCodeBasicInfo &GCBI,
            TupleTree<model::Binary> &Binary,
            FunctionSummaryOracle &Oracle,
            CFGAnalyzer &Analyzer) :
    M(M),
    Context(M.getContext()),
    GCBI(GCBI),
    Binary(Binary),
    Oracle(Oracle),
    Analyzer(Analyzer),
    EntryHook(getEntryHookType(M), "entry_hook", &M) {}

public:
  static llvm::FunctionType *getEntryHookType(llvm::Module &M) {
    auto Int8PtrTy = llvm::Type::getInt8PtrTy(M.getContext());
    auto VoidTy = llvm::Type::getVoidTy(M.getContext());
    return llvm::FunctionType::get(VoidTy, { Int8PtrTy }, false);
  }

  void run();

private:
  void outlineFunction(model::Function &F);
  void saveFinalResults(InterproceduralNode *Node, const IA::Results &);
  void computeApproximateCallGraph();
  void initializeInterproceduralQueue();
  std::pair<OutlinedFunction, ABIAnalyses::ABIAnalysesResults>
  analyzeABI(llvm::BasicBlock *Entry);
  void runIntraproceduralAnalysis();
  InterproceduralGraph constructInterproceduralGraph();
  void runInterproceduralAnalysis();
  void runMFPAnalysis();
  void interproceduralPropagation();
  void finalizeModel();
  void applyABIDeductions();

private:
  CSVSet computePreservedCSVs(const CSVSet &ClobberedRegisters) const;

  TrackingSortedVector<model::Register::Values>
  computePreservedRegisters(const CSVSet &ClobberedRegisters) const;

  CSVSet findWrittenRegisters(llvm::Function *F);

  UpcastablePointer<model::Type>
  buildPrototypeForIndirectCall(const FunctionSummary &CallerSummary,
                                const efa::BasicBlock &CallerBlock,
                                const ABISummary &ABI);

  // TODO: Concept for both T and Inserter types
  template<typename T, typename Inserter>
  void insertTypedRegister(const llvm::GlobalVariable *CSV, Inserter &I) {
    auto Register = model::Register::fromCSVName(CSV->getName(),
                                                 Binary->Architecture());

    if (Register != model::Register::Invalid and CSV != GCBI.spReg()) {
      const auto Type = CSV->getValueType();
      const auto Size = Type->getIntegerBitWidth() / 8;

      T TR(Register);
      TR.Type() = {
        Binary->getPrimitiveType(model::PrimitiveTypeKind::Generic, Size), {}
      };
      I.insert(TR);
    }
  }
};

void DetectABI::initializeInterproceduralQueue() {

  // Create an over-approximated call graph of the program. A queue of all the
  // function entrypoints is maintained.
  for (auto *Node : llvm::post_order(&ApproximateCallGraph)) {
    // Ignore entry node
    if (Node == ApproximateCallGraph.getEntryNode())
      continue;

    // The intraprocedural analysis will be scheduled only for those functions
    // which have `Invalid` as type.
    auto &Function = Binary->Functions().at(Node->Address);
    EntrypointsQueue.insert(Node);
  }

  revng_assert(Binary->Functions().size() == EntrypointsQueue.size());
}

void DetectABI::computeApproximateCallGraph() {
  using llvm::BasicBlock;

  using BasicBlockToNodeMap = llvm::DenseMap<BasicBlock *, BasicBlockNode *>;
  BasicBlockToNodeMap BasicBlockNodeMap;

  // Temporary worklist to collect the function entrypoints
  llvm::SmallVector<BasicBlock *, 8> Worklist;

  // Create an over-approximated call graph
  for (const auto &Function : Binary->Functions()) {
    auto *Entry = GCBI.getBlockAt(Function.Entry());
    BasicBlockNode Node{ Function.Entry() };
    BasicBlockNode *GraphNode = ApproximateCallGraph.addNode(Node);
    BasicBlockNodeMap[Entry] = GraphNode;
  }

  for (const auto &Function : Binary->Functions()) {
    llvm::SmallSet<BasicBlock *, 8> Visited;
    auto *Entry = GCBI.getBlockAt(Function.Entry());
    revng_assert(Entry != nullptr);

    BasicBlockNode *StartNode = BasicBlockNodeMap[Entry];
    revng_assert(StartNode != nullptr);
    Worklist.emplace_back(Entry);

    while (!Worklist.empty()) {
      BasicBlock *Current = Worklist.pop_back_val();
      Visited.insert(Current);

      if (hasMarker(Current, "function_call")) {
        // If not an indirect call, add the node to the CG
        if (BasicBlock *Callee = getFunctionCallCallee(Current)) {
          BasicBlockNode *GraphNode = nullptr;
          auto It = BasicBlockNodeMap.find(Callee);
          if (It != BasicBlockNodeMap.end()) {
            StartNode->addSuccessor(It->second);
          }
        }
        BasicBlock *Next = getFallthrough(Current);
        revng_assert(Next != nullptr);

        if (Visited.count(Next) == 0)
          Worklist.push_back(Next);

      } else {

        for (BasicBlock *Successor : successors(Current)) {
          if (not isPartOfRootDispatcher(Successor)
              && !Visited.count(Successor)) {
            revng_assert(Successor != nullptr);
            Worklist.push_back(Successor);
          }
        }
      }
    }
  }

  // Create a root entry node for the call-graph, connect all the nodes to it,
  // and perform a post-order traversal. Keep in mind that adding a root node
  // as a predecessor to all nodes does not affect POT of any node, except the
  // root node itself.
  BasicBlockNode *RootNode = ApproximateCallGraph.addNode(MetaAddress());
  ApproximateCallGraph.setEntryNode(RootNode);

  for (const auto &[_, Node] : BasicBlockNodeMap)
    RootNode->addSuccessor(Node);

  // Dump the call-graph, if requested
  if (CallGraphOutputPath.getNumOccurrences() == 1) {
    std::ifstream File(CallGraphOutputPath.c_str());
    std::error_code EC;
    raw_fd_ostream OutputCG(CallGraphOutputPath, EC);
    revng_assert(!EC);
    llvm::WriteGraph(OutputCG, &ApproximateCallGraph);
  }
}

UpcastablePointer<model::Type>
DetectABI::buildPrototypeForIndirectCall(const FunctionSummary &CallerSummary,
                                         const efa::BasicBlock &CallerBlock,
                                         const ABISummary &ABI) {
  using namespace model;

  auto NewType = makeType<RawFunctionType>();
  auto &CallType = *llvm::cast<RawFunctionType>(NewType.get());
  {
    auto ArgumentsInserter = CallType.Arguments().batch_insert();
    auto ReturnValuesInserter = CallType.ReturnValues().batch_insert();

    using std::ranges::for_each;
    for_each(ABI.Arguments, [&](auto *CSV) {
      insertTypedRegister<NamedTypedRegister>(CSV, ArgumentsInserter);
    });
    for_each(ABI.Returns, [&](auto *CSV) {
      insertTypedRegister<TypedRegister>(CSV, ReturnValuesInserter);
    });

    // Import FinalStackOffset and CalleeSavedRegisters from the default
    // prototype
    const FunctionSummary &DefaultSummary = Oracle.getDefault();
    const auto &Clobbered = DefaultSummary.ClobberedRegisters;
    CallType.PreservedRegisters() = computePreservedRegisters(Clobbered);

    CallType.FinalStackOffset() = DefaultSummary.ElectedFSO.value_or(0);
  }

  return NewType;
}

/// Finish the population of the model by building the prototype
void DetectABI::finalizeModel() {
  using namespace model;

  // Fill up the model and build its prototype for each function
  std::set<model::Function *> Functions;
  for (model::Function &Function : Binary->Functions()) {
    // Ignore if we already have a prototype
    if (Function.Prototype().isValid())
      continue;

    MetaAddress EntryPC = Function.Entry();
    revng_assert(EntryPC.isValid());
    auto &OutlinedFunction = OutlinedFunctions[EntryPC];
    auto Summary = Analyzer.analyze(&OutlinedFunction);

    // Replace function attributes
    Function.Attributes() = Summary.Attributes;

    auto NewType = makeType<RawFunctionType>();
    auto &FunctionType = *llvm::cast<RawFunctionType>(NewType.get());
    {
      auto ArgumentsInserter = FunctionType.Arguments().batch_insert();
      auto ReturnValuesInserter = FunctionType.ReturnValues().batch_insert();

      auto &ABI = FinalResults[EntryPC];

      using std::ranges::for_each;
      for_each(ABI.Arguments, [&](auto *CSV) {
        insertTypedRegister<NamedTypedRegister>(CSV, ArgumentsInserter);
      });
      for_each(ABI.Returns, [&](auto *CSV) {
        insertTypedRegister<TypedRegister>(CSV, ReturnValuesInserter);
      });

      // Preserved registers
      const auto &ClobberedRegisters = Summary.ClobberedRegisters;
      auto PreservedRegisters = computePreservedRegisters(ClobberedRegisters);
      FunctionType.PreservedRegisters() = std::move(PreservedRegisters);

      // Final stack offset
      FunctionType.FinalStackOffset() = Summary.ElectedFSO.value_or(0);
    }

    Function.Prototype() = Binary->recordNewType(std::move(NewType));
    Functions.insert(&Function);

    Oracle.getLocalFunction(EntryPC) = std::move(Summary);
  }

  // Build prototype for indirect function calls
  for (auto &Function : Functions) {
    auto &Summary = Oracle.getLocalFunction(Function->Entry());
    for (auto &Block : Summary.CFG) {

      // TODO: we do not detect prototypes for inlined call sites
      if (Block.ID().isInlined())
        continue;

      MetaAddress BlockAddress = Block.ID().notInlinedAddress();

      Node::CallSiteAddress Key{ Function->Entry(), BlockAddress };
      const auto &ABI = FinalResults[Key];

      for (auto &Edge : Block.Successors()) {
        if (auto *CE = llvm::dyn_cast<efa::CallEdge>(Edge.get())) {
          auto &CallSitePrototypes = Function->CallSitePrototypes();
          bool IsDirect = CE->Destination().isValid();
          bool IsDynamic = not CE->DynamicFunction().empty();
          bool HasInfoOnEdge = CallSitePrototypes.count(BlockAddress) != 0;
          if (not IsDynamic and not IsDirect and not HasInfoOnEdge) {
            // It's an indirect call for which we have now call site information

            auto Prototype = buildPrototypeForIndirectCall(Summary, Block, ABI);
            auto Path = Binary->recordNewType(std::move(Prototype));

            // This analysis does not have the power to detect whether an
            // indirect call site is a tail call, noreturn or inline
            bool IsTailCall = false;

            // Register new prototype
            model::CallSitePrototype ThePrototype(BlockAddress,
                                                  Path,
                                                  IsTailCall,
                                                  {});
            Function->CallSitePrototypes().insert(std::move(ThePrototype));
          }
        }
      }
    }

    efa::FunctionMetadata FM(Function->Entry(), Summary.CFG);
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
void DetectABI::interproceduralPropagation() {
  for (const model::Function &Function : Binary->Functions()) {
    // TODO: get ABIResults from different source
    auto &Summary = Oracle.getLocalFunction(Function.Entry());
    for (auto &[PC, CallSite] : Summary.ABIResults.CallSites) {

      if (PC.isInlined())
        continue;

      if (PC.notInlinedAddress() == Function.Entry())
        combineCrossCallSites(CallSite, Summary.ABIResults);
    }
  }
}

using MaybeRegisterState = std::optional<abi::RegisterState::Values>;
using ABIAnalyses::RegisterStateMap;

MaybeRegisterState
DetectABI::tryGetRegisterState(model::Register::Values RegisterValue,
                               const RegisterStateMap &ABIRegisterMap) {
  using State = abi::RegisterState::Values;

  llvm::StringRef Name = model::Register::getCSVName(RegisterValue);
  if (llvm::GlobalVariable *CSV = M.getGlobalVariable(Name, true)) {
    auto It = ABIRegisterMap.find(CSV);
    if (It != ABIRegisterMap.end()) {
      revng_assert(It->second != State::Count && It->second != State::Invalid);
      return It->second;
    }
  }

  return std::nullopt;
}

void DetectABI::initializeMapForDeductions(FunctionSummary &Summary,
                                           abi::RegisterState::Map &Map) {
  auto Arch = model::ABI::getArchitecture(Binary->DefaultABI());
  revng_assert(Arch == Binary->Architecture());

  for (const auto &Reg : model::Architecture::registers(Arch)) {
    const auto &ArgRegisters = Summary.ABIResults.ArgumentsRegisters;
    const auto &RVRegisters = Summary.ABIResults.FinalReturnValuesRegisters;

    if (auto MaybeState = tryGetRegisterState(Reg, ArgRegisters))
      Map[Reg].IsUsedForPassingArguments = *MaybeState;

    if (auto MaybeState = tryGetRegisterState(Reg, RVRegisters))
      Map[Reg].IsUsedForReturningValues = *MaybeState;
  }
}

void DetectABI::applyABIDeductions() {
  using namespace abi;

  if (ABIEnforcement == NoABIEnforcement)
    return;

  for (const model::Function &Function : Binary->Functions()) {
    auto &Summary = Oracle.getLocalFunction(Function.Entry());

    RegisterState::Map StateMap(Binary->Architecture());
    initializeMapForDeductions(Summary, StateMap);

    bool EnforceABIConformance = ABIEnforcement == FullABIEnforcement;
    std::optional<abi::RegisterState::Map> ResultMap;

    // TODO: drop this.
    // Since function type conversion is capable of handling the holes
    // internally, there's not much reason to push such invasive changes
    // this early in the pipeline.
    auto ABI = abi::Definition::get(Binary->DefaultABI());
    if (EnforceABIConformance)
      ResultMap = ABI.enforceRegisterState(StateMap);
    else
      ResultMap = ABI.tryDeducingRegisterState(StateMap);

    if (!ResultMap.has_value())
      continue;

    for (const auto &[Register, State] : *ResultMap) {
      llvm::StringRef Name = model::Register::getCSVName(Register);
      if (llvm::GlobalVariable *CSV = M.getGlobalVariable(Name, true)) {
        auto MaybeArg = State.IsUsedForPassingArguments;
        auto MaybeRV = State.IsUsedForReturningValues;

        // ABI-refined results per function
        if (Summary.ABIResults.ArgumentsRegisters.count(CSV) != 0)
          Summary.ABIResults.ArgumentsRegisters[CSV] = MaybeArg;

        if (Summary.ABIResults.FinalReturnValuesRegisters.count(CSV) != 0)
          Summary.ABIResults.FinalReturnValuesRegisters[CSV] = MaybeRV;

        // ABI-refined results per indirect call-site
        for (auto &Block : Summary.CFG) {
          for (auto &Edge : Block.Successors()) {
            if (efa::FunctionEdgeType::isCall(Edge->Type())
                && Edge->Type() != efa::FunctionEdgeType::FunctionCall) {
              revng_assert(Block.ID().isValid());
              auto &CSSummary = Summary.ABIResults.CallSites.at(Block.ID());

              if (CSSummary.ArgumentsRegisters.count(CSV) != 0)
                CSSummary.ArgumentsRegisters[CSV] = MaybeArg;

              if (CSSummary.ReturnValuesRegisters.count(CSV) != 0)
                CSSummary.ReturnValuesRegisters[CSV] = MaybeRV;
            }
          }
        }
      }
    }

    if (Log.isEnabled()) {
      Log << "Summary for " << Function.OriginalName() << ":\n";
      Summary.dump(Log);
      Log << DoLog;
    }
  }
}

std::set<llvm::GlobalVariable *>
DetectABI::findWrittenRegisters(llvm::Function *F) {
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

std::set<llvm::GlobalVariable *>
DetectABI::computePreservedCSVs(const CSVSet &ClobberedRegisters) const {
  using llvm::GlobalVariable;
  using std::set;
  set<GlobalVariable *> PreservedRegisters(Analyzer.abiCSVs().begin(),
                                           Analyzer.abiCSVs().end());
  std::erase_if(PreservedRegisters, [&](const auto &E) {
    auto End = ClobberedRegisters.end();
    return ClobberedRegisters.find(E) != End;
  });

  return PreservedRegisters;
}

TrackingSortedVector<model::Register::Values>
DetectABI::computePreservedRegisters(const CSVSet &ClobberedRegisters) const {
  using namespace model;

  TrackingSortedVector<model::Register::Values> Result;
  CSVSet PreservedRegisters = computePreservedCSVs(ClobberedRegisters);

  auto PreservedRegistersInserter = Result.batch_insert();
  for (auto *CSV : PreservedRegisters) {
    auto RegisterID = model::Register::fromCSVName(CSV->getName(),
                                                   Binary->Architecture());
    if (RegisterID == Register::Invalid)
      continue;

    PreservedRegistersInserter.insert(RegisterID);
  }

  return Result;
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

std::pair<OutlinedFunction, ABIAnalyses::ABIAnalysesResults>
DetectABI::analyzeABI(llvm::BasicBlock *Entry) {
  using namespace llvm;
  using llvm::BasicBlock;
  using namespace ABIAnalyses;

  MetaAddress EntryAddress = getBasicBlockAddress(Entry);

  IRBuilder<> Builder(M.getContext());
  ABIAnalysesResults ABIResults;

  // Detect function boundaries
  OutlinedFunction OutlinedFunction = Analyzer.outline(Entry);

  // Find registers that may be target of at least one store. This helps
  // refine the final results.
  auto WrittenRegisters = findWrittenRegisters(OutlinedFunction.Function.get());

  // Run ABI-independent data-flow analyses
  ABIResults = analyzeOutlinedFunction(OutlinedFunction.Function.get(),
                                       GCBI,
                                       Analyzer.preCallHook(),
                                       Analyzer.postCallHook(),
                                       Analyzer.retHook());

  // We say that a register is callee-saved when, besides being preserved by
  // the callee, there is at least a write onto this register.
  const auto &Summary = Oracle.getLocalFunction(EntryAddress);
  auto CalleeSavedRegs = computePreservedCSVs(Summary.ClobberedRegisters);
  auto ActualCalleeSavedRegs = intersect(CalleeSavedRegs, WrittenRegisters);

  // Union between effective callee-saved registers and SP
  ActualCalleeSavedRegs.insert(GCBI.spReg());

  // Refine ABI analyses results by suppressing callee-saved and stack
  // pointer registers.
  suppressCSAndSPRegisters(ABIResults, ActualCalleeSavedRegs);

  // Merge return values registers
  ABIAnalyses::finalizeReturnValues(ABIResults);

  return {std::move(OutlinedFunction), ABIResults};
}

void DetectABI::runInterproceduralAnalysis() {
  std::set<MetaAddress> Set;

  while (!EntrypointsQueue.empty()) {
    const BasicBlockNode *EntryNode = EntrypointsQueue.pop();
    MetaAddress EntryPointAddress = EntryNode->Address;
    revng_log(Log, "Analyzing Entry: " << EntryPointAddress.toString());
    LoggerIndent<> Indent(Log);

    // Intraprocedural analysis

    // TODO: here we are interested in 1) being noreturn or not,
    //       2) callee-saved registers and 3) FSO.
    //       However, `analyze` also computes the CFG. There's a refactoring
    //       opportunity.
    llvm::BasicBlock *BB = GCBI.getBlockAt(EntryNode->Address);
    OutlinedFunction OutlinedFunction = Analyzer.outline(BB);
    FunctionSummary AnalysisResult = Analyzer.analyze(&OutlinedFunction);

    if (Log.isEnabled()) {
      AnalysisResult.dump(Log);
      Log << DoLog;
    }

    // Serialize CFG in root
    {
      efa::FunctionMetadata New;
      New.Entry() = EntryNode->Address;
      New.ControlFlowGraph() = AnalysisResult.CFG;
      New.simplify(*Binary);
      New.serialize(GCBI);
    }

    // Perform some early sanity checks once the CFG is ready
    revng_assert(AnalysisResult.CFG.size() > 0);
    for (const MetaAddress &MA : Set)
      revng_assert(Oracle.getLocalFunction(MA).CFG.size() > 0);

    if (Binary->Functions()[EntryPointAddress].Prototype().isValid())
      continue;

    bool Changed = Oracle.registerLocalFunction(EntryPointAddress,
                                                std::move(AnalysisResult));

    Set.insert(EntryPointAddress);

    // If we got improved results for a function, we need to recompute its
    // callers, and if a caller turns out to be an inline function, the callers
    // of the inline function too.
    if (Changed) {
      revng_log(Log,
                "Entry " << EntryPointAddress.toString() << " has changed");
      LoggerIndent<> Indent(Log);
      OnceQueue<const BasicBlockNode *> InlineFunctionWorklist;
      InlineFunctionWorklist.insert(EntryNode);

      while (!InlineFunctionWorklist.empty()) {
        const BasicBlockNode *Node = InlineFunctionWorklist.pop();
        MetaAddress NodeAddress = Node->Address;
        revng_log(Log,
                  "Re-enqueuing callers of " << NodeAddress.toString() << ":");
        LoggerIndent<> Indent(Log);
        for (auto *Caller : Node->predecessors()) {
          // Root node?
          if (Caller->Address.isInvalid())
            continue;

          // If it's inline, re-enqueue its callers too
          MetaAddress CallerPC = Caller->Address;
          const auto &CallerSummary = Oracle.getLocalFunction(CallerPC);
          using namespace model::FunctionAttribute;
          bool IsInline = CallerSummary.Attributes.count(Inline) != 0;
          if (IsInline)
            InlineFunctionWorklist.insert(Caller);

          if (not Binary->Functions().at(CallerPC).Prototype().isValid()) {
            revng_log(Log, CallerPC.toString());
            EntrypointsQueue.insert(Caller);
          }
        }
      }
    }
  }
}

bool DetectABIPass::runOnModule(Module &M) {
  revng_log(PassesLog, "Starting EarlyFunctionAnalysis");

  if (not M.getFunction("root") or M.getFunction("root")->isDeclaration())
    return false;

  auto &GCBI = getAnalysis<GeneratedCodeBasicInfoWrapperPass>().getGCBI();
  auto &LMP = getAnalysis<LoadModelWrapperPass>().get();

  TupleTree<model::Binary> &Binary = LMP.getWriteableModel();

  FunctionSummaryOracle Oracle;
  importModel(M, GCBI, *Binary, Oracle);

  CFGAnalyzer Analyzer(M, GCBI, Binary, Oracle);

  DetectABI ABIDetector(M, GCBI, Binary, Oracle, Analyzer);

  ABIDetector.run();

  return false;
}

void DetectABI::outlineFunction(model::Function &F) {
  // Outline function and save it for future use
  MetaAddress Addr = F.Entry();
  llvm::BasicBlock *Entry = GCBI.getBlockAt(Addr);
  OutlinedFunctions[Addr] = Analyzer.outline(Entry);

  // Create @entry_hook call at the beginning of entry block
  auto &OEntry = OutlinedFunctions[Addr].Function->getEntryBlock();
  IRBuilder<> Builder(&*OEntry.getFirstInsertionPt());
  Builder.CreateCall(EntryHook.get(), { Addr.toValue(&M) });
}

void DetectABI::saveFinalResults(InterproceduralNode *Node,
                                 const IA::Results &MFPResults) {

  revng_assert(Node->isResult());

  const bool IsIndirect = Node->isIndirect();

  // Use OutValue from ...
  const auto &ResultLattice = MFPResults.OutValue;
  const llvm::BasicBlock *Entry = Node->BB;

  // Identify entry in FinalResults by address:
  //  - MetaAddress of Entry for functions
  //  - (MetaAddress Entry, MetaAddress CallSite) for call sites
  using RT = InterproceduralNode::ResultType;
  auto &Results = FinalResults[Node->Address];

  // Depending on type of Results (Arguments or Returns) we save information
  // from LatticeElement to appropriate set.
  auto &RegisterSet = (Node->getResultType() == RT::Arguments) ?
                        Results.Arguments :
                        Results.Returns;

  const InterproceduralAnalysis::LatticeElement::RegisterSet
    &LatticeSet = (Node->getResultType() == RT::Arguments) ?
                    ResultLattice.Arguments :
                    ResultLattice.Returns;

  auto GetReg =
    [this](const llvm::GlobalVariable *GV) -> model::Register::Values {
    return model::Register::fromCSVName(GV->getName(), Binary->Architecture());
  };

  // Keep all registers from results for Indirect calls. Indirect prototypes
  // will be built later.
  if (IsIndirect) {
    RegisterSet = LatticeSet;
  } else {
    const auto &FunctionAddress = Node->getFunctionAddress();
    auto &Summary = Oracle.getLocalFunction(FunctionAddress);

    auto *Function = OutlinedFunctions[FunctionAddress].Function.get();
    auto WrittenRegisters = findWrittenRegisters(Function);
    auto CSR = computePreservedCSVs(Summary.ClobberedRegisters);
    auto ActualCalleeSaved = intersect(WrittenRegisters, CSR);
    ActualCalleeSaved.insert(GCBI.spReg());

    auto IsNotPreserved =
      [&ActualCalleeSaved](const llvm::GlobalVariable *GV) -> bool {
      llvm::GlobalVariable *NGV = const_cast<llvm::GlobalVariable *>(GV);
      return ActualCalleeSaved.find(NGV) == ActualCalleeSaved.end();
    };

    using std::ranges::copy_if;
    copy_if(LatticeSet,
            std::inserter(RegisterSet, RegisterSet.begin()),
            IsNotPreserved);
  }
}

void DetectABI::runMFPAnalysis() {
  // OutlineFunctions
  InterproceduralAnalysis IA(GCBI,
                             Binary,
                             PartialResults,
                             EntryHook.get(),
                             Analyzer.preCallHook(),
                             Analyzer.postCallHook(),
                             Analyzer.retHook());
  for_each(Binary->Functions(), [this](auto F) { outlineFunction(F); });

  // Construct
  auto G = constructInterproceduralGraph();
  InterproceduralAnalysis::LatticeElement InitialValue{};
  InterproceduralAnalysis::LatticeElement ExtremalValue{ true };
  auto Start = G.getEntryNode();

  revng_assert(G.size() != 0);
  revng_assert(Start != nullptr);

  auto Results = MFP::getMaximalFixedPoint(IA,
                                           Start,
                                           InitialValue,
                                           ExtremalValue,
                                           { Start },
                                           { Start });

  for (auto &[Node, MFPResult] : Results) {
    if (Node->isResult()) {
      saveFinalResults(Node, MFPResult);
    }
  }
}

void DetectABI::runIntraproceduralAnalysis() {
  using std::ranges::for_each;

  for_each(Binary->Functions(), [this](model::Function F) {
    llvm::BasicBlock *Entry = GCBI.getBlockAt(F.Entry());
    auto [OutlinedFunction, Results] = analyzeABI(Entry);
    OutlinedFunctions[F.Entry()] = std::move(OutlinedFunction);
    PartialResults[F.Entry()] = Results;
  });
}

void DetectABI::run() {
  using std::ranges::for_each;

  computeApproximateCallGraph();

  initializeInterproceduralQueue();

  runInterproceduralAnalysis();

  runIntraproceduralAnalysis();

  runMFPAnalysis();

  // Commit the results onto the model. A non-const model is taken as
  // argument to be written.
  finalizeModel();
}

InterproceduralGraph DetectABI::constructInterproceduralGraph() {
  using std::tuple;
  using Key = tuple<Node::AddressType, const llvm::BasicBlock *, Node::VType>;
  std::map<Key, ForwardNode<Node> *> Map;

  InterproceduralGraph G;

  auto *EntryNode = G.addNode();
  G.setEntryNode(EntryNode);

  auto GetNode = [&Map, &G, EntryNode](Node::AddressType Address,
                                       const llvm::BasicBlock *BB,
                                       Node::VType V) -> ForwardNode<Node> * {
    auto It = Map.find(Key{ Address, BB, V });
    if (It != Map.end()) {
      return It->second;
    }

    ForwardNode<Node> *Node = G.addNode(Address, BB, V);
    Map.emplace(Key{ Address, BB, V }, Node);

    if (std::holds_alternative<Node::AnalysisType>(V)) {
      EntryNode->addSuccessor(Node);
    }

    return Node;
  };

  auto GetAddressFromOperand = [](const llvm::Instruction *I,
                                  unsigned int N) -> MetaAddress {
    auto Operand = dyn_cast<llvm::CallInst>(I)->getArgOperand(N);
    if (Operand) {
      return BasicBlockID::fromValue(Operand).start();
    }

    return MetaAddress{};
  };

  for (auto &[CallerEntry, F] : OutlinedFunctions) {
    auto &Function = *F.Function;

    auto *EntryBB = &Function.getEntryBlock();
    auto *CallerArgs = GetNode(CallerEntry,
                               EntryBB,
                               Node::ResultType::Arguments);
    auto *CallerRets = GetNode(CallerEntry, EntryBB, Node::ResultType::Returns);

    for (auto &BB : Function) {
      using ABIAnalyses::getPreCallHook;
      using ABIAnalyses::isCallSiteBlock;

      if (auto PreCallHook = getPreCallHook(&BB);
          PreCallHook != nullptr) { // BB = PreCallHook
        auto PreCallHookInst = dyn_cast<llvm::CallInst>(PreCallHook);

        auto Callee = PreCallHookInst->getArgOperand(1);
        auto CalleeAddress = BasicBlockID::fromValue(Callee).start();

        auto CallSite = PreCallHookInst->getArgOperand(0);
        auto CallSiteAddress = BasicBlockID::fromValue(CallSite).start();

        revng_assert(CallSiteAddress.isValid());

        auto *CallSiteBlock = &BB;
        revng_assert(isCallSiteBlock(CallSiteBlock));
        revng_assert(CallSiteBlock->getUniquePredecessor() != nullptr);

        // UAOF
        // +------------------+
        // |    C.Argument    |
        // +------------------+
        //   |
        //   | ∀C, C∈F.Callees
        //   v
        // #==================#
        // H     UAOF(F)      H
        // #==================#
        //   |
        //   |
        //   v
        // +------------------+
        // |    F.Argument    |
        // +------------------+
        auto CalleeArgs = GetNode(CalleeAddress,
                                  &BB,
                                  Node::ResultType::Arguments);
        auto CallerUAOF = GetNode(CallerEntry,
                                  EntryBB,
                                  Node::AnalysisType::UAOF);
        CalleeArgs->addSuccessor(CallerUAOF);
        CallerUAOF->addSuccessor(CallerArgs);

        // RAOFC
        //                                             +---------------------+
        //                                             | CS1.Caller.Argument |
        //                                             +---------------------+
        //                                               |
        //            ∀CS, CS∈CS1.PreviousCallSites      |
        //                                   |           v
        // +-----------------------+         |         #=====================#
        // | CS.Callee.ReturnValue | ----------------> H     RAOFC(CS1)      H
        // +-----------------------+                   #=====================#
        //                                               |
        //                                               |
        //                                               v
        //                                             +---------------------+
        //                                             | CS1.Callee.Argument |
        //                                             +---------------------+
        auto *RAOFC = GetNode(Node::CallSiteAddress{ CallerEntry,
                                                     CallSiteAddress },
                              CallSiteBlock,
                              Node::AnalysisType::RAOFC);
        CallerArgs->addSuccessor(RAOFC);
        RAOFC->addSuccessor(CalleeArgs);

        // TODO: (document this) Previous CallSites (RAOFC);
        if (auto *Start = BB.getUniquePredecessor(); Start != nullptr) {
          for (llvm::BasicBlock *PrevBB : llvm::inverse_depth_first(Start)) {
            if (auto *PrevPreCallHook = getPreCallHook(PrevBB);
                PrevPreCallHook != nullptr) {
              // TODO: comment this
              auto Addr = GetAddressFromOperand(PrevPreCallHook, 1);
              auto Node = GetNode(Node::CallSiteAddress{ CallerEntry, Addr },
                                  PrevBB,
                                  Node::ResultType::Returns);

              Node->addSuccessor(RAOFC);
            }
          }
        }

        // URVOF
        // +------------------+
        // |  C.ReturnValue   |
        // +------------------+
        //   |
        //   | ∀C, C∈F.Callees
        //   v
        // #==================#
        // H     URVOF(F)     H
        // #==================#
        //   |
        //   |
        //   v
        // +------------------+
        // |  F.ReturnValue   |
        // +------------------+
        auto *URVOF = GetNode(CallerEntry, &BB, Node::AnalysisType::URVOF);
        auto *CalleeRets = GetNode(CalleeAddress,
                                   &BB,
                                   Node::ResultType::Returns);
        CalleeRets->addSuccessor(URVOF);
        URVOF->addSuccessor(CallerRets);

        // URVOFC
        //                                        +-----------------------+
        //                                       | F1.Callee.ReturnValue
        //                                       |
        //                                       +-----------------------+
        //                                         |
        //          ∀CS, CS∈F1.NextCallSites       |
        //                             |           v
        //+--------------------+       | #=======================# |
        // CS.Callee.Argument | --------------> H      URVOFC(F1)       H
        //+--------------------+ #=======================#
        //                                         |
        //                                         |
        //                                         v
        //                                       +-----------------------+
        //                                       | F1.Caller.ReturnValue
        //                                       |
        //                                       +-----------------------+
        //
        auto *URVOFC = GetNode(Node::CallSiteAddress{ CallerEntry,
                                                      CallSiteAddress },
                               CallSiteBlock,
                               Node::AnalysisType::URVOFC);

        CalleeRets->addSuccessor(URVOFC);
        URVOFC->addSuccessor(CallerRets);

        // TODO: (document this) next callsites (URVOFC)
        if (auto *Start = BB.getUniqueSuccessor(); Start != nullptr) {
          for (llvm::BasicBlock *NextBB : llvm::depth_first(Start)) {
            if (auto *NextPreCallHook = getPreCallHook(NextBB);
                NextPreCallHook != nullptr) {
              auto Addr = GetAddressFromOperand(NextPreCallHook, 1);
              auto Node = GetNode(Node::CallSiteAddress{ CallerEntry, Addr },
                                  NextBB,
                                  Node::ResultType::Arguments);

              Node->addSuccessor(URVOFC);
            }
          }
        }
      }
    }
  }

  std::error_code EC;
  llvm::raw_fd_ostream OutputGraph("/tmp/graph.dot", EC);
  revng_assert(!EC);
  llvm::WriteGraph(OutputGraph, &G);

  return G;
}

DetectABI::CSVSet DetectABI::findWrittenRegisters(llvm::Function *F) {
  using namespace llvm;

  DetectABI::CSVSet WrittenRegisters;
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

char DetectABIPass::ID = 0;

using ABIDetectionPass = RegisterPass<DetectABIPass>;
static ABIDetectionPass X("detect-abi", "ABI Detection Pass", true, false);

} // namespace efa
