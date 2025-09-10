/// \file DetectABI.cpp

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <fstream>
#include <iterator>
#include <memory>

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetOperations.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/GraphWriter.h"

#include "revng/ABI/Definition.h"
#include "revng/ABI/FunctionType/Layout.h"
#include "revng/ADT/Queue.h"
#include "revng/BasicAnalyses/GeneratedCodeBasicInfo.h"
#include "revng/EarlyFunctionAnalysis/CFGAnalyzer.h"
#include "revng/EarlyFunctionAnalysis/CallEdge.h"
#include "revng/EarlyFunctionAnalysis/CallGraph.h"
#include "revng/EarlyFunctionAnalysis/CollectFunctionsFromCalleesPass.h"
#include "revng/EarlyFunctionAnalysis/CollectFunctionsFromUnusedAddressesPass.h"
#include "revng/EarlyFunctionAnalysis/ControlFlowGraph.h"
#include "revng/EarlyFunctionAnalysis/ControlFlowGraphCache.h"
#include "revng/EarlyFunctionAnalysis/DetectABI.h"
#include "revng/EarlyFunctionAnalysis/FunctionEdgeBase.h"
#include "revng/EarlyFunctionAnalysis/FunctionSummaryOracle.h"
#include "revng/Model/Binary.h"
#include "revng/Model/NameBuilder.h"
#include "revng/Model/Pass/DeduplicateCollidingNames.h"
#include "revng/Model/Register.h"
#include "revng/Pipeline/Pipe.h"
#include "revng/Pipeline/RegisterAnalysis.h"
#include "revng/Pipes/Kinds.h"
#include "revng/Pipes/ModelGlobal.h"
#include "revng/Support/BasicBlockID.h"
#include "revng/Support/Debug.h"
#include "revng/Support/IRHelpers.h"
#include "revng/Support/MetaAddress.h"
#include "revng/Support/OpaqueRegisterUser.h"

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

struct Changes {
  bool Function = false;
  std::set<MetaAddress> Callees;
};

class DetectABIAnalysis {
public:
  static constexpr auto Name = "detect-abi";

  std::vector<std::vector<pipeline::Kind *>> AcceptedKinds = {
    { &revng::kinds::Root }
  };

  void run(const pipeline::ExecutionContext &EC,
           pipeline::LLVMContainer &ModuleContainer) {
    revng::pipes::CFGMap CFGs("");
    llvm::legacy::PassManager Manager;
    using namespace revng;
    auto Global = cantFail(EC.getContext()
                             .getGlobal<ModelGlobal>(ModelGlobalName));
    Manager.add(new LoadModelWrapperPass(ModelWrapper(Global->get())));
    Manager.add(new CollectFunctionsFromCalleesWrapperPass());
    Manager.add(new ControlFlowGraphCachePass(CFGs));
    Manager.add(new efa::DetectABIPass());
    Manager.add(new CollectFunctionsFromUnusedAddressesWrapperPass());
    Manager.add(new efa::DetectABIPass());
    Manager.run(ModuleContainer.getModule());
  }
};

static pipeline::RegisterAnalysis<DetectABIAnalysis> A1;

namespace efa {

static model::Architecture::Values getCodeArchitecture(const MetaAddress &MA) {
  const auto MaybeArch = MetaAddressType::arch(MA.type());
  revng_assert(MaybeArch && "The architecture is available for code addresses");
  return model::Architecture::fromLLVMArchitecture(*MaybeArch);
}

static bool isWritingToMemory(llvm::Instruction &I) {
  if (auto *StoreInst = dyn_cast<llvm::StoreInst>(&I)) {
    auto *Destination = StoreInst->getPointerOperand();

    return isMemory(Destination);
  }

  return false;
}

using BasicBlockQueue = UniquedQueue<const BasicBlockNode *>;

class DetectABI {
private:
  using BasicBlockToNodeMap = llvm::DenseMap<llvm::BasicBlock *,
                                             BasicBlockNode *>;

private:
  llvm::Module &M;
  llvm::LLVMContext &Context;
  GeneratedCodeBasicInfo &GCBI;
  ControlFlowGraphCache &FMC;
  TupleTree<model::Binary> &Binary;
  FunctionSummaryOracle &Oracle;
  CFGAnalyzer &Analyzer;

  CallGraph ApproximateCallGraph;
  BasicBlockToNodeMap BasicBlockNodeMap;

public:
  DetectABI(llvm::Module &M,
            GeneratedCodeBasicInfo &GCBI,
            ControlFlowGraphCache &FMC,
            TupleTree<model::Binary> &Binary,
            FunctionSummaryOracle &Oracle,
            CFGAnalyzer &Analyzer) :
    M(M),
    Context(M.getContext()),
    GCBI(GCBI),
    FMC(FMC),
    Binary(Binary),
    Oracle(Oracle),
    Analyzer(Analyzer) {}

public:
  void run() {
    llvm::Task Task(6, "DetectABI");
    Task.advance("computeApproximateCallGraph");
    computeApproximateCallGraph();

    // Perform a preliminary analysis of the function
    //
    // We're interested:
    //
    // 1. the function being noreturn or not;
    // 2. the list of callee-saved registers;
    // 3. the final stack offset;
    // 4. the CFG (specifically, which indirect jumps are returns);
    Task.advance("preliminaryFunctionAnalysis");
    preliminaryFunctionAnalysis();

    // Run the (fixed-point) analysis of the ABI of each function
    Task.advance("analyzeABI");
    analyzeABI();

    // Refine results with ABI-specific information
    Task.advance("applyABIDeductions");
    applyABIDeductions();

    // Commit the results onto the model. A non-const model is taken as
    // argument to be written.
    Task.advance("finalizeModel");
    finalizeModel();

    // Propagate prototypes
    Task.advance("propagatePrototypes");
    propagatePrototypes();
  }

private:
  void computeApproximateCallGraph();
  void preliminaryFunctionAnalysis();
  void analyzeABI();
  Changes analyzeFunctionABI(const model::Function &Function,
                             OutlinedFunction &OutlinedFunction,
                             OpaqueRegisterUser &Clobberer);
  void applyABIDeductions();

  /// Finish the population of the model by building the prototype
  void finalizeModel();

  // Calls propagatePrototypesInFunction for all functions in Binary
  void propagatePrototypes();

  /// Propagate prototypes to callers
  void propagatePrototypesInFunction(model::Function &Function);

private:
  void recordRegisters(const efa::CSVSet &CSVs, auto Inserter);

  CSVSet computePreservedCSVs(const CSVSet &ClobberedRegisters) const;

  TrackingSortedVector<model::Register::Values>
  computePreservedRegisters(const CSVSet &ClobberedRegisters) const;

  Changes runAnalyses(MetaAddress EntryAddress,
                      OutlinedFunction &OutlinedFunction);

  CSVSet findWrittenRegisters(llvm::Function *F);

  model::UpcastableType
  buildPrototypeForIndirectCall(const FunctionSummary &CallerSummary,
                                const efa::BasicBlock &CallerBlock);

  bool getRegisterState(model::Register::Values, const CSVSet &);
};

void DetectABI::computeApproximateCallGraph() {
  using llvm::BasicBlock;

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
            auto IsCurrentSuccessor = [&It](auto &Successor) {
              return Successor == It->second;
            };
            if (not any_of(StartNode->successors(), IsCurrentSuccessor))
              StartNode->addSuccessor(It->second);
          }
        }
        BasicBlock *Next = getFallthrough(Current);
        revng_assert(Next != nullptr);

        if (!Visited.contains(Next))
          Worklist.push_back(Next);

      } else {

        for (BasicBlock *Successor : successors(Current)) {
          if (not isPartOfRootDispatcher(Successor)
              && !Visited.contains(Successor)) {
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
    std::error_code EC;
    raw_fd_ostream OutputCG(CallGraphOutputPath, EC);
    revng_assert(!EC);
    llvm::WriteGraph(OutputCG, &ApproximateCallGraph);
  }
}

void DetectABI::preliminaryFunctionAnalysis() {
  revng_log(Log, "Running the preliminary function analysis");
  LoggerIndent<> LodIndent(Log);

  BasicBlockQueue EntrypointsQueue;

  //
  // Populate queue of entry points
  //

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

  //
  // Process the queue
  //

  while (!EntrypointsQueue.empty()) {
    const BasicBlockNode *EntryNode = EntrypointsQueue.pop();
    MetaAddress EntryPointAddress = EntryNode->Address;
    revng_log(Log, "Analyzing " << EntryPointAddress.toString());
    LoggerIndent<> Indent(Log);

    FunctionSummary AnalysisResult = Analyzer.analyze(EntryNode->Address);

    if (Log.isEnabled()) {
      AnalysisResult.dump(Log);
      Log << DoLog;
    }

    // Serialize CFG in the ControlFlowGraphCache
    {
      TupleTree<efa::ControlFlowGraph> New;
      New->Entry() = EntryNode->Address;
      New->Blocks() = AnalysisResult.CFG;
      New->simplify(*Binary);
      FMC.set(std::move(New));
    }

    // Perform some early sanity checks once the CFG is ready
    revng_assert(AnalysisResult.CFG.size() > 0);

    if (not Binary->Functions()[EntryPointAddress].Prototype().isEmpty()) {
      Oracle.getLocalFunction(EntryPointAddress)
        .CFG = std::move(AnalysisResult.CFG);
      continue;
    }

    bool Changed = Oracle.registerLocalFunction(EntryPointAddress,
                                                std::move(AnalysisResult));

    revng_assert(Oracle.getLocalFunction(EntryPointAddress).CFG.size() > 0);

    // If we got improved results for a function, we need to recompute its
    // callers, and if a caller turns out to be an inline function, the
    // callers of the inline function too.
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
          if (CallerSummary.Attributes.contains(Inline))
            InlineFunctionWorklist.insert(Caller);

          if (Binary->Functions().at(CallerPC).Prototype().isEmpty()) {
            revng_log(Log, CallerPC.toString());
            EntrypointsQueue.insert(Caller);
          }
        }
      }
    }
  }

  // Register all indirect call sites in all functions, we'll need to fill in
  // their prototypes
  for (auto &Function : Binary->Functions()) {
    auto &CFG = Oracle.getLocalFunction(Function.Entry()).CFG;
    revng_log(Log,
              "Registering indirect call sites of "
                << Function.Entry().toString());
    for (efa::BasicBlock &Block : CFG) {
      for (auto &Edge : Block.Successors()) {
        if (auto *Call = dyn_cast<CallEdge>(Edge.get())) {
          if (Call->isIndirect()) {
            revng_log(Log,
                      "Registering " << Function.Entry().toString() << " "
                                     << Block.ID().toString());
            Oracle.registerCallSite(Function.Entry(),
                                    Block.ID(),
                                    FunctionSummary(),
                                    Call->IsTailCall());
          }
        }
      }
    }
  }
}

void DetectABI::analyzeABI() {
  revng_log(Log, "Running ABI analyses");
  LoggerIndent<> Indent(Log);

  llvm::Task Task(2, "analyzeABI");
  std::map<MetaAddress, std::unique_ptr<OutlinedFunction>> Functions;

  // Create all temporary functions
  Task.advance("Create temporary functions");
  for (model::Function &Function : Binary->Functions()) {
    const MetaAddress &Entry = Function.Entry();
    auto NewFunction = make_unique<OutlinedFunction>(Analyzer.outline(Entry));
    Functions[Function.Entry()] = std::move(NewFunction);
  }

  // Push this into analyzeFunction
  OpaqueRegisterUser RegisterUser(&M);

  // TODO: this really needs to become a monotone framework
  Task.advance("Run fixed-point analyses");
  llvm::Task FixedPointTask({}, "Fixed-point analysis");
  UniquedQueue<model::Function *> ToAnalyze;
  for (model::Function &Function : Binary->Functions())
    ToAnalyze.insert(&Function);

  // Change the oracle default prototype to have no arguments nor return values
  {
    auto NewDefault = Oracle.getDefault().clone();
    NewDefault.ABIResults.ArgumentsRegisters = {};
    NewDefault.ABIResults.ReturnValuesRegisters = {};
    Oracle.setDefault(std::move(NewDefault));
  }

  unsigned Runs = 0;
  model::CNameBuilder NameBuilder = *Binary;
  while (not ToAnalyze.empty()) {
    model::Function &Function = *ToAnalyze.pop();
    revng_log(Log, "Analyzing " << Function.Entry().toString());
    FixedPointTask.advance(NameBuilder.name(Function));
    OutlinedFunction &OutlinedFunction = *Functions.at(Function.Entry());
    Changes Changes = analyzeFunctionABI(Function,
                                         OutlinedFunction,
                                         RegisterUser);

    if (Changes.Function) {
      revng_log(Log, "The function has changed, re-enqueing all callers:");
      LoggerIndent<> Indent(Log);
      // The prototype of the function we analyzed has changed, reanalyze
      // callers
      auto &FunctionNode = BasicBlockNodeMap[GCBI.getBlockAt(Function.Entry())];
      for (auto &CallerNode : FunctionNode->predecessors()) {
        if (CallerNode->Address.isValid()) {
          revng_log(Log, CallerNode->Address.toString());
          ToAnalyze.insert(&Binary->Functions().at(CallerNode->Address));
        }
      }
    }

    // Register for re-analysis all the callees for which we have new
    // information
    for (const MetaAddress &ToReanalyze : Changes.Callees) {
      revng_assert(ToReanalyze.isValid());
      revng_log(Log, "Re-enqueing callee " << ToReanalyze.toString());
      ToAnalyze.insert(&Binary->Functions().at(ToReanalyze));
    }
  }
}

Changes DetectABI::analyzeFunctionABI(const model::Function &Function,
                                      OutlinedFunction &OutlinedFunction,
                                      OpaqueRegisterUser &RegisterReader) {
  revng_log(Log, "Analyzing the ABI of " << Function.Entry().toString());

  //
  // Enrich all the call sites
  //

  // Collect all calls to precall_hook and postcall_hook
  SmallVector<std::pair<CallInst *, bool>> Hooks;
  for (Instruction &I : instructions(OutlinedFunction.Function.get())) {
    if (CallInst *CallToPreHook = getCallTo(&I, Analyzer.preCallHook())) {
      Hooks.emplace_back(CallToPreHook, true);
    } else if (CallInst *CallToPostHook = getCallTo(&I,
                                                    Analyzer.postCallHook())) {
      Hooks.emplace_back(CallToPostHook, false);
    }
  }

  IRBuilder<> Builder(M.getContext());
  for (auto &[Call, IsPreHook] : Hooks) {
    auto CallerBlock = BasicBlockID::fromValue(Call->getArgOperand(0));
    auto CalleeAddress = MetaAddress::fromValue(Call->getArgOperand(1));
    auto ExtractString = extractFromConstantStringPtr;
    StringRef CalleeSymbol = ExtractString(Call->getArgOperand(2));
    auto &&[Summary, _] = Oracle.getCallSite(Function.Entry(),
                                             CallerBlock,
                                             CalleeAddress,
                                             CalleeSymbol);
    auto &ABIResults = Summary->ABIResults;

    auto *BB = Call->getParent();
    if (IsPreHook) {
      Builder.SetInsertPoint(Call);
      for (auto *CSV : ABIResults.ArgumentsRegisters) {
        // Inject a virtual read of the arguments of the callee
        // TODO: drop const_cast. Unfortunately it requires a significant
        //       refactoring.
        RegisterReader.read(Builder, const_cast<GlobalVariable *>(CSV));
      }

      // Ensure the precall_hook is the first instruction of the block
      BB->splitBasicBlockBefore(Call);
    } else {
      // Ensure the postcall_hook is the last instruction of the block
      BB->splitBasicBlockBefore(Call->getNextNode());

      Builder.SetInsertPoint(BB, BB->getFirstInsertionPt());
      for (auto *CSV : ABIResults.ReturnValuesRegisters) {
        // Inject a virtual write to the return values of the callee
        // TODO: drop const_cast. Unfortunately it requires a significant
        //       refactoring.
        RegisterReader.write(Builder, const_cast<GlobalVariable *>(CSV));
      }
    }
  }

  // Perform the analysis
  Changes Changes = runAnalyses(Function.Entry(), OutlinedFunction);

  RegisterReader.purgeCreated();

  return Changes;
}

// TODO: drop this.
// Since function type conversion is capable of handling the holes
// internally, there's not much reason to push such invasive changes
// this early in the pipeline.
void DetectABI::applyABIDeductions() {
  if (ABIEnforcement == NoABIEnforcement)
    return;

  auto ABI = abi::Definition::get(Binary->DefaultABI());
  for (const model::Function &Function : Binary->Functions()) {
    auto &Summary = Oracle.getLocalFunction(Function.Entry());

    abi::Definition::RegisterSet Arguments;
    abi::Definition::RegisterSet RValues;
    model::Architecture::Values Architecture = Binary->Architecture();
    for (const auto &Register : model::Architecture::registers(Architecture)) {
      llvm::StringRef Name = model::Register::getCSVName(Register);
      if (llvm::GlobalVariable *CSV = M.getGlobalVariable(Name, true)) {
        if (Summary.ABIResults.ArgumentsRegisters.contains(CSV))
          Arguments.emplace(Register);

        if (Summary.ABIResults.ReturnValuesRegisters.contains(CSV))
          RValues.emplace(Register);
      }
    }

    if (ABIEnforcement == FullABIEnforcement) {
      Arguments = ABI.enforceArgumentRegisterState(std::move(Arguments));
      RValues = ABI.enforceReturnValueRegisterState(std::move(RValues));
    } else {
      if (auto R = ABI.tryDeducingArgumentRegisterState(std::move(Arguments)))
        Arguments = *R;
      else
        continue; // Register deduction failed.

      if (auto R = ABI.tryDeducingReturnValueRegisterState(std::move(RValues)))
        RValues = *R;
      else
        continue; // Register deduction failed.
    }

    efa::CSVSet ResultingArguments;
    efa::CSVSet ResultingReturnValues;
    for (const auto &Register : model::Architecture::registers(Architecture)) {
      llvm::StringRef Name = model::Register::getCSVName(Register);
      if (llvm::GlobalVariable *CSV = M.getGlobalVariable(Name, true)) {
        if (Arguments.contains(Register))
          ResultingArguments.insert(CSV);
        if (RValues.contains(Register))
          ResultingReturnValues.insert(CSV);
      }
    }

    for (auto &Block : Summary.CFG) {
      for (auto &Edge : Block.Successors()) {
        if (efa::FunctionEdgeType::isCall(Edge->Type())
            && Edge->Type() != efa::FunctionEdgeType::FunctionCall) {
          revng_assert(Block.ID().isValid());
          auto &CSSummary = Summary.ABIResults.CallSites.at(Block.ID());
          CSSummary.ArgumentsRegisters = ResultingArguments;
          CSSummary.ReturnValuesRegisters = ResultingReturnValues;
        }
      }
    }
    Summary.ABIResults.ArgumentsRegisters = std::move(ResultingArguments);
    Summary.ABIResults.ReturnValuesRegisters = std::move(ResultingReturnValues);

    if (Log.isEnabled()) {
      Log << "Summary for " << Function.Name() << ":\n";
      Summary.dump(Log);
      Log << DoLog;
    }
  }
}

void DetectABI::recordRegisters(const efa::CSVSet &CSVs, auto Inserter) {
  for (auto *CSV : CSVs) {
    auto Reg = model::Register::fromCSVName(CSV->getName(),
                                            Binary->Architecture());
    Inserter.emplace(Reg).Type() = model::PrimitiveType::makeGeneric(Reg);
  }
}

void DetectABI::finalizeModel() {
  using namespace model;

  // Fill up the model and build its prototype for each function
  std::set<model::Function *> Functions;
  for (model::Function &Function : Binary->Functions()) {
    // Ignore if we already have a prototype
    if (not Function.Prototype().isEmpty())
      continue;

    MetaAddress EntryPC = Function.Entry();
    revng_assert(EntryPC.isValid());
    auto &Summary = Oracle.getLocalFunction(EntryPC);

    // Replace function attributes
    Function.Attributes() = Summary.Attributes;

    auto &&[Prototype, NewType] = Binary->makeRawFunctionDefinition();
    Prototype.Architecture() = getCodeArchitecture(EntryPC);

    // Record arguments and return values
    recordRegisters(Summary.ABIResults.ArgumentsRegisters,
                    Prototype.Arguments().batch_insert());
    recordRegisters(Summary.ABIResults.ReturnValuesRegisters,
                    Prototype.ReturnValues().batch_insert());

    // Preserved registers
    const auto &ClobberedRegisters = Summary.ClobberedRegisters;
    auto PreservedRegisters = computePreservedRegisters(ClobberedRegisters);
    Prototype.PreservedRegisters() = std::move(PreservedRegisters);

    // Final stack offset
    Prototype.FinalStackOffset() = Summary.ElectedFSO.value_or(0);

    Function.Prototype() = std::move(NewType);
    Functions.insert(&Function);
  }

  // Build prototype for indirect function calls
  for (auto &Function : Functions) {
    auto &Summary = Oracle.getLocalFunction(Function->Entry());
    for (auto &Block : Summary.CFG) {

      // TODO: we do not detect prototypes for inlined call sites
      if (Block.ID().isInlined())
        continue;
      MetaAddress BlockAddress = Block.ID().notInlinedAddress();

      for (auto &Edge : Block.Successors()) {
        if (auto *CE = llvm::dyn_cast<efa::CallEdge>(Edge.get())) {
          auto &CallSitePrototypes = Function->CallSitePrototypes();
          bool IsDirect = CE->Destination().isValid();
          bool IsDynamic = not CE->DynamicFunction().empty();
          bool HasInfoOnEdge = CallSitePrototypes.contains(BlockAddress);
          if (not IsDynamic and not IsDirect and not HasInfoOnEdge) {
            // It's an indirect call for which we have now call site information
            auto Prototype = buildPrototypeForIndirectCall(Summary, Block);

            // This analysis does not have the power to detect whether an
            // indirect call site is a tail call, noreturn or inline
            bool IsTailCall = false;

            // Register new prototype
            Function->CallSitePrototypes().emplace(BlockAddress,
                                                   std::move(Prototype),
                                                   IsTailCall);
          }
        }
      }
    }
  }

  for (auto &Function : Functions) {
    auto &Summary = Oracle.getLocalFunction(Function->Entry());
    efa::ControlFlowGraph FM(Function->Entry(), "", Summary.CFG);
    FM.verify(*Binary, true);
  }

  revng_check(Binary->verify(true));
}

void DetectABI::propagatePrototypes() {
  for (model::Function &Function : Binary->Functions()) {
    propagatePrototypesInFunction(Function);
  }
}

// TODO: is this still necessary after the new EFA?
void DetectABI::propagatePrototypesInFunction(model::Function &Function) {
  const MetaAddress &Entry = Function.Entry();

  revng_log(Log, "Trying to propagate prototypes for " << Entry.toString());
  LoggerIndent<> Indent(Log);

  FunctionSummary &Summary = Oracle.getLocalFunction(Entry);
  RUAResults &ABI = Summary.ABIResults;
  SortedVector<efa::BasicBlock> &CFG = Summary.CFG;
  CSVSet &WrittenRegisters = Summary.WrittenRegisters;

  bool IsSingleNode = CFG.size() == 1;

  revng_log(Log, "IsSingleNode: " << IsSingleNode);

  if (not IsSingleNode)
    return;

  efa::BasicBlock &Block = *CFG.begin();
  bool HasSingleSuccessor = Block.Successors().size() == 1;
  revng_log(Log, "HasSingleSuccessor: " << HasSingleSuccessor);

  if (not HasSingleSuccessor)
    return;

  auto &Successor = *Block.Successors().begin();
  const auto *Call = dyn_cast<efa::CallEdge>(Successor.get());
  bool SuccessorIsCall = Call != nullptr;
  revng_log(Log, "SuccessorIsCall: " << SuccessorIsCall);

  // Select new prototype for wrapper function
  if (SuccessorIsCall) {
    auto Prototype = getPrototype(*Binary, Entry, Block, *Call);

    using abi::FunctionType::Layout;
    // Get layout of wrapped function
    Layout CalleeLayout = Layout::make(*Prototype);

    // Verify that wrapper function:
    //  - don't write stack pointer
    //  - don't write to callee arguments
    //  - every store instruction writes to registers (not memory)
    llvm::BasicBlock *BB = GCBI.getBlockAt(Block.ID().start());

    GlobalVariable *StackPointer = GCBI.spReg();

    // Check if wrapper writes to Stack Pointer
    const bool WritesSP = WrittenRegisters.contains(StackPointer);

    using std::ranges::count_if;
    auto IsWrittenByCaller = [this, &WrittenRegisters](auto &Argument) {
      auto *CSV = M.getGlobalVariable(model::Register::getName(Argument));
      return WrittenRegisters.count(CSV);
    };
    const auto &Arguments = CalleeLayout.argumentRegisters();
    bool WritesCalleeArgs = any_of(Arguments, IsWrittenByCaller);
    const auto &ReturnValues = CalleeLayout.returnValueRegisters();
    bool WritesCalleeReturnValues = any_of(ReturnValues, IsWrittenByCaller);

    bool WritesToMemory = count_if(*BB, isWritingToMemory) > 0;
    bool WritesOnlyRegisters = WritesToMemory == 0;

    revng_log(Log, "WritesSP: " << WritesSP);
    revng_log(Log, "WritesCalleeArgs: " << WritesCalleeArgs);
    revng_log(Log, "WritesOnlyRegisters: " << WritesOnlyRegisters);

    // When above conditions are met, overwrite wrappers prototype with
    // wrapped function prototype (`CABIFunctionDefinition` or
    // `RawFunctionDefinition`)
    if (not WritesSP and not WritesCalleeArgs and not WritesCalleeReturnValues
        and WritesOnlyRegisters) {

      if (Log.isEnabled()) {
        Log << "Overwriting " << Entry.toString() << " prototype ";
        if (!Function.Prototype().isEmpty())
          Log << "(" << toString(Function.Prototype()) << ") ";
        Log << "with wrapped function's prototype: "
            << toString(model::copyTypeDefinition(*Prototype)) << DoLog;
      }

      Function.Prototype() = Binary->makeType(Prototype->key());

      if (Function.Name().empty()) {
        if (not Call->DynamicFunction().empty()) {
          Function.Name() = Call->DynamicFunction();

        } else if (Call->Destination().isValid()) {
          const model::Function &Callee = Binary->Functions()
                                            .at(Call->Destination()
                                                  .notInlinedAddress());
          if (not Callee.Name().empty())
            Function.Name() = Callee.Name();
          else if (not Callee.Name().empty())
            Function.Name() = Callee.Name();
        }
      }
    }
  }

  // TODO: should this be done at a higher abstraction level?
  model::deduplicateCollidingNames(Binary);
}

model::UpcastableType
DetectABI::buildPrototypeForIndirectCall(const FunctionSummary &CallerSummary,
                                         const efa::BasicBlock &CallerBlock) {
  using namespace model;

  auto &&[Prototype, NewType] = Binary->makeRawFunctionDefinition();
  Prototype.Architecture() = getCodeArchitecture(CallerBlock.ID().start());

  bool Found = false;
  for (const auto &[PC, CallSites] : CallerSummary.ABIResults.CallSites) {
    if (PC != CallerBlock.ID())
      continue;

    revng_assert(!Found);
    Found = true;

    recordRegisters(CallSites.ArgumentsRegisters,
                    Prototype.Arguments().batch_insert());
    recordRegisters(CallSites.ReturnValuesRegisters,
                    Prototype.ReturnValues().batch_insert());
  }
  revng_assert(Found);

  // Import FinalStackOffset and CalleeSavedRegisters from the default
  // prototype
  const FunctionSummary &DefaultSummary = Oracle.getDefault();
  const auto &Clobbered = DefaultSummary.ClobberedRegisters;
  Prototype.PreservedRegisters() = computePreservedRegisters(Clobbered);

  Prototype.FinalStackOffset() = DefaultSummary.ElectedFSO.value_or(0);

  return std::move(NewType);
}

static void combineCrossCallSites(auto &CallSite, auto &Callee) {
  // TODO: why not return values?
  for (auto *CSV : CallSite.ArgumentsRegisters) {
    Callee.ArgumentsRegisters.insert(CSV);
  }
}

bool DetectABI::getRegisterState(model::Register::Values RegisterValue,
                                 const CSVSet &ABIRegisterMap) {

  llvm::StringRef Name = model::Register::getCSVName(RegisterValue);
  if (llvm::GlobalVariable *CSV = M.getGlobalVariable(Name, true)) {
    return ABIRegisterMap.contains(CSV);
  }

  return false;
}

CSVSet DetectABI::findWrittenRegisters(llvm::Function *F) {
  using namespace llvm;

  CSVSet WrittenRegisters;
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

CSVSet DetectABI::computePreservedCSVs(const CSVSet &ClobberedRegisters) const {
  using llvm::GlobalVariable;
  using std::set;
  CSVSet PreservedRegisters(Analyzer.abiCSVs().begin(),
                            Analyzer.abiCSVs().end());

  for (GlobalVariable *ClobberedRegister : ClobberedRegisters)
    PreservedRegisters.erase(ClobberedRegister);

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

static void suppressCalleeSaved(RUAResults &ABIResults,
                                const CSVSet &CalleeSavedRegs) {

  // Suppress from arguments
  for (const auto &Reg : CalleeSavedRegs)
    ABIResults.ArgumentsRegisters.erase(Reg);

  // Suppress from return values
  for (const auto &Reg : CalleeSavedRegs)
    ABIResults.ReturnValuesRegisters.erase(Reg);

  // Suppress from call-sites
  for (const auto &[K, _] : ABIResults.CallSites) {
    for (const auto &Reg : CalleeSavedRegs) {
      ABIResults.CallSites[K].ArgumentsRegisters.erase(Reg);
      ABIResults.CallSites[K].ReturnValuesRegisters.erase(Reg);
    }
  }
}

template<typename T, typename F>
static auto zeroOrOne(const T &Range, const F &Predicate)
  -> decltype(&*Range.begin()) {
  decltype(&*Range.begin()) Result = nullptr;
  for (auto &E : Range) {
    if (Predicate(E)) {
      revng_assert(not Result);
      Result = &E;
    }
  }

  return Result;
}

Changes DetectABI::runAnalyses(MetaAddress EntryAddress,
                               OutlinedFunction &OutlinedFunction) {
  using namespace llvm;
  using llvm::BasicBlock;

  IRBuilder<> Builder(M.getContext());
  RUAResults ABIResults;

  // Find registers that may be target of at least one store. This helps
  // refine the final results.
  auto WrittenRegisters = findWrittenRegisters(OutlinedFunction.Function.get());

  // Run ABI-independent data-flow analyses
  ABIResults = analyzeRegisterUsage(OutlinedFunction.Function.get(),
                                    GCBI,
                                    Binary->Architecture(),
                                    Analyzer.preCallHook(),
                                    Analyzer.postCallHook(),
                                    Analyzer.retHook());

  // We say that a register is callee-saved when, besides being preserved by
  // the callee, there is at least a write onto this register.
  FunctionSummary &Summary = Oracle.getLocalFunction(EntryAddress);
  auto CalleeSavedRegs = computePreservedCSVs(Summary.ClobberedRegisters);
  auto ActualCalleeSavedRegs = llvm::set_intersection(CalleeSavedRegs,
                                                      WrittenRegisters);

  // Refine ABI analyses results by suppressing callee-saved and stack
  // pointer registers.
  suppressCalleeSaved(ABIResults, ActualCalleeSavedRegs);

  // Commit ABI analysis results to the oracle
  auto Old = Summary.ABIResults;
  Summary.ABIResults.combine(ABIResults);
  Summary.WrittenRegisters = WrittenRegisters;

  Changes Changes;
  Changes.Function = Old != Summary.ABIResults;

  for (auto &[BlockID, CallSite] : ABIResults.CallSites) {
    // TODO: why are we ignoring inlined call sites?
    if (BlockID.isInlined())
      continue;

    MetaAddress Callee = CallSite.CalleeAddress;

    // TODO: eventually we'll want to add arguments/return values to dynamic
    //       functions too
    FunctionSummary *Summary = nullptr;
    if (Callee.isValid()) {
      Summary = &Oracle.getLocalFunction(Callee);
      revng_assert(Summary != nullptr);
    } else {
      Summary = Oracle.getExactCallSite(EntryAddress, BlockID).first;
    }

    // TODO: are longjmps calls? Do we want to collect prototype for them?
    //       Right now, they are in the IR, but we do not register them as call
    //       sites in the Oracle.
    if (Summary != nullptr) {
      bool Changed = false;
      RUAResults &ToAdjust = Summary->ABIResults;

      for (auto *CSV : CallSite.ArgumentsRegisters)
        Changed = Changed or ToAdjust.ArgumentsRegisters.insert(CSV).second;

      for (auto *CSV : CallSite.ReturnValuesRegisters)
        Changed = Changed or ToAdjust.ReturnValuesRegisters.insert(CSV).second;

      if (Changed and Callee.isValid())
        Changes.Callees.insert(Callee);
    }
  }

  return Changes;
}

bool DetectABIPass::runOnModule(Module &M) {
  revng_log(PassesLog, "Starting EarlyFunctionAnalysis");

  if (not M.getFunction("root") or M.getFunction("root")->isDeclaration())
    return false;

  auto &GCBI = getAnalysis<GeneratedCodeBasicInfoWrapperPass>().getGCBI();
  auto &FMC = getAnalysis<ControlFlowGraphCachePass>().get();
  auto &LMP = getAnalysis<LoadModelWrapperPass>().get();

  TupleTree<model::Binary> &Binary = LMP.getWriteableModel();

  using FSOracle = FunctionSummaryOracle;
  FSOracle Oracle = FSOracle::importFullPrototypes(M, GCBI, *Binary);
  CFGAnalyzer Analyzer(M, GCBI, Binary, Oracle);

  DetectABI ABIDetector(M, GCBI, FMC, Binary, Oracle, Analyzer);

  ABIDetector.run();

  return false;
}

char DetectABIPass::ID = 0;

using ABIDetectionPass = RegisterPass<DetectABIPass>;
static ABIDetectionPass X("detect-abi", "ABI Detection Pass", true, false);

} // namespace efa
