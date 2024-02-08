/// \file DetectABI.cpp

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <fstream>
#include <memory>

#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/GraphWriter.h"

#include "revng/ABI/Definition.h"
#include "revng/ABI/FunctionType/Layout.h"
#include "revng/ABI/RegisterState.h"
#include "revng/ADT/Queue.h"
#include "revng/BasicAnalyses/GeneratedCodeBasicInfo.h"
#include "revng/EarlyFunctionAnalysis/CFGAnalyzer.h"
#include "revng/EarlyFunctionAnalysis/CallGraph.h"
#include "revng/EarlyFunctionAnalysis/CollectFunctionsFromCalleesPass.h"
#include "revng/EarlyFunctionAnalysis/CollectFunctionsFromUnusedAddressesPass.h"
#include "revng/EarlyFunctionAnalysis/DetectABI.h"
#include "revng/EarlyFunctionAnalysis/FunctionMetadata.h"
#include "revng/EarlyFunctionAnalysis/FunctionSummaryOracle.h"
#include "revng/Model/Binary.h"
#include "revng/Model/Pass/PromoteOriginalName.h"
#include "revng/Model/Register.h"
#include "revng/Pipeline/Pipe.h"
#include "revng/Pipeline/RegisterAnalysis.h"
#include "revng/Pipes/Kinds.h"
#include "revng/Pipes/LLVMAnalysisImplementation.h"
#include "revng/Support/BasicBlockID.h"
#include "revng/Support/Debug.h"
#include "revng/Support/IRHelpers.h"
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
  static constexpr auto Name = "detect-abi";

  std::vector<std::vector<pipeline::Kind *>> AcceptedKinds = {
    { &revng::kinds::Root }
  };

  void print(const pipeline::Context &Ctx,
             llvm::raw_ostream &OS,
             llvm::ArrayRef<std::string> ContainerNames) const {
    Implementation.print(Ctx, OS, ContainerNames);
  }

  void run(const pipeline::ExecutionContext &Ctx,
           pipeline::LLVMContainer &Container) {
    Implementation.run(Ctx, Container);
  }
};

static pipeline::RegisterAnalysis<DetectABIAnalysis> A1;

namespace efa {

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
  using CSVSet = std::set<llvm::GlobalVariable *>;

private:
  llvm::Module &M;
  llvm::LLVMContext &Context;
  GeneratedCodeBasicInfo &GCBI;
  TupleTree<model::Binary> &Binary;
  FunctionSummaryOracle &Oracle;
  CFGAnalyzer &Analyzer;

  CallGraph ApproximateCallGraph;

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
    Analyzer(Analyzer) {}

public:
  void run() {
    computeApproximateCallGraph();

    // Perform a preliminary analysis of the function
    //
    // We're interested:
    //
    // 1. the function being noreturn or not;
    // 2. the list of callee-saved registers;
    // 3. the final stack offset;
    // 4. the CFG (specifically, which indirect jumps are returns);
    preliminaryFunctionAnalysis();

    // Run the (fixed-point) analysis of the ABI of each function
    analyzeABI();

    // Refine results with ABI-specific information
    applyABIDeductions();

    // Commit the results onto the model. A non-const model is taken as
    // argument to be written.
    finalizeModel();

    // Propagate prototypes
    propagatePrototypes();
  }

private:
  void computeApproximateCallGraph();
  void preliminaryFunctionAnalysis();
  void analyzeABI();
  bool analyzeFunctionABI(const model::Function &Function,
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
  CSVSet computePreservedCSVs(const CSVSet &ClobberedRegisters) const;

  TrackingSortedVector<model::Register::Values>
  computePreservedRegisters(const CSVSet &ClobberedRegisters) const;

  bool runAnalyses(MetaAddress EntryAddress,
                   OutlinedFunction &OutlinedFunction);

  CSVSet findWrittenRegisters(llvm::Function *F);

  UpcastablePointer<model::Type>
  buildPrototypeForIndirectCall(const FunctionSummary &CallerSummary,
                                const efa::BasicBlock &CallerBlock);

  std::optional<abi::RegisterState::Values>
  tryGetRegisterState(model::Register::Values,
                      const ABIAnalyses::RegisterStateMap &);

  void initializeMapForDeductions(FunctionSummary &, abi::RegisterState::Map &);
};

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
    std::ifstream File(CallGraphOutputPath.c_str());
    std::error_code EC;
    raw_fd_ostream OutputCG(CallGraphOutputPath, EC);
    revng_assert(!EC);
    llvm::WriteGraph(OutputCG, &ApproximateCallGraph);
  }
}

void DetectABI::preliminaryFunctionAnalysis() {
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
    revng_log(Log, "Analyzing Entry: " << EntryPointAddress.toString());
    LoggerIndent<> Indent(Log);

    llvm::BasicBlock *BB = GCBI.getBlockAt(EntryNode->Address);
    FunctionSummary AnalysisResult = Analyzer.analyze(BB);

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

    if (not Binary->Functions()[EntryPointAddress].Prototype().empty()) {
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

          if (Binary->Functions().at(CallerPC).Prototype().empty()) {
            revng_log(Log, CallerPC.toString());
            EntrypointsQueue.insert(Caller);
          }
        }
      }
    }
  }
}

void DetectABI::analyzeABI() {
  std::map<MetaAddress, std::unique_ptr<OutlinedFunction>> Functions;

  // Create all temporary functions
  for (model::Function &Function : Binary->Functions()) {
    llvm::BasicBlock *Entry = GCBI.getBlockAt(Function.Entry());
    auto NewFunction = make_unique<OutlinedFunction>(Analyzer.outline(Entry));
    Functions[Function.Entry()] = std::move(NewFunction);
  }

  // Push this into analyzeFunction
  OpaqueRegisterUser RegisterReader(&M);

  // TODO: this really needs to become a monotone framework
  bool Changed = true;
  unsigned Runs = 0;
  while (Changed) {
    ++Runs;
    revng_log(Log, "Run #" << Runs);
    LoggerIndent<> Indent(Log);

    Changed = false;

    for (model::Function &Function : Binary->Functions()) {
      OutlinedFunction &OutlinedFunction = *Functions.at(Function.Entry());
      Changed = analyzeFunctionABI(Function, OutlinedFunction, RegisterReader)
                or Changed;
    }
  }
}

bool DetectABI::analyzeFunctionABI(const model::Function &Function,
                                   OutlinedFunction &OutlinedFunction,
                                   OpaqueRegisterUser &RegisterReader) {
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

  //
  for (auto &[Call, IsPreHook] : Hooks) {
    auto CallerBlock = BasicBlockID::fromValue(Call->getArgOperand(0));
    auto CalleeAddress = MetaAddress::fromValue(Call->getArgOperand(1));
    auto ExtractString = extractFromConstantStringPtr;
    StringRef CalleeSymbol = ExtractString(Call->getArgOperand(2));
    auto [Summary, _] = Oracle.getCallSite(Function.Entry(),
                                           CallerBlock,
                                           CalleeAddress,
                                           CalleeSymbol);
    auto &ABIResults = Summary->ABIResults;

    IRBuilder<> Builder(M.getContext());
    if (IsPreHook) {
      Builder.SetInsertPoint(Call);
      for (auto &[Register, Value] : ABIResults.ArgumentsRegisters) {
        if (Value == abi::RegisterState::Yes) {
          // Inject a virtual read of the arguments of the callee

          // Note: injecting this is detrimental for RAOFC, it prevents it from
          //       detecting an argument of a call site. However, this is not a
          //       problem, since we already marked the current register as an
          //       argument of the all site, and we only add (never remove)
          //       registers to the set of arguments.
          // TODO: we should also do this for return values of the function
          // TODO: drop const_cast. Unfortunately it requires a significant
          //       refactoring.
          RegisterReader.read(Builder, const_cast<GlobalVariable *>(Register));
        }
      }

      // Ensure the precall_hook is the first instruction of the block
      Call->getParent()->splitBasicBlockBefore(Call);
    } else {
      // Ensure the postcall_hook is the last instruction of the block
      Call->getParent()->splitBasicBlockBefore(Call->getNextNode());
    }
  }

  // Perform the analysis
  bool Result = runAnalyses(Function.Entry(), OutlinedFunction);

  RegisterReader.purgeCreated();

  return Result;
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
        if (Summary.ABIResults.ArgumentsRegisters.contains(CSV))
          Summary.ABIResults.ArgumentsRegisters[CSV] = MaybeArg;

        if (Summary.ABIResults.FinalReturnValuesRegisters.contains(CSV))
          Summary.ABIResults.FinalReturnValuesRegisters[CSV] = MaybeRV;

        // ABI-refined results per indirect call-site
        for (auto &Block : Summary.CFG) {
          for (auto &Edge : Block.Successors()) {
            if (efa::FunctionEdgeType::isCall(Edge->Type())
                && Edge->Type() != efa::FunctionEdgeType::FunctionCall) {
              revng_assert(Block.ID().isValid());
              auto &CSSummary = Summary.ABIResults.CallSites.at(Block.ID());

              if (CSSummary.ArgumentsRegisters.contains(CSV))
                CSSummary.ArgumentsRegisters[CSV] = MaybeArg;

              if (CSSummary.ReturnValuesRegisters.contains(CSV))
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

void DetectABI::finalizeModel() {
  using namespace model;
  using RegisterState = abi::RegisterState::Values;

  // Fill up the model and build its prototype for each function
  std::set<model::Function *> Functions;
  for (model::Function &Function : Binary->Functions()) {
    // Ignore if we already have a prototype
    if (not Function.Prototype().empty())
      continue;

    MetaAddress EntryPC = Function.Entry();
    revng_assert(EntryPC.isValid());
    auto &Summary = Oracle.getLocalFunction(EntryPC);

    // Replace function attributes
    Function.Attributes() = Summary.Attributes;

    auto NewType = makeType<RawFunctionType>();
    auto &FunctionType = *llvm::cast<RawFunctionType>(NewType.get());
    {
      auto ArgumentsInserter = FunctionType.Arguments().batch_insert();
      auto ReturnValuesInserter = FunctionType.ReturnValues().batch_insert();

      // Argument and return values
      for (const auto &[Arg, RV] :
           zipmap_range(Summary.ABIResults.ArgumentsRegisters,
                        Summary.ABIResults.FinalReturnValuesRegisters)) {
        auto *CSV = Arg == nullptr ? RV->first : Arg->first;
        RegisterState RSArg = Arg == nullptr ? RegisterState::Maybe :
                                               Arg->second;
        RegisterState RSRV = RV == nullptr ? RegisterState::Maybe : RV->second;

        auto RegisterID = model::Register::fromCSVName(CSV->getName(),
                                                       Binary->Architecture());
        if (RegisterID == Register::Invalid || CSV == GCBI.spReg())
          continue;

        auto *CSVType = CSV->getValueType();
        auto CSVSize = CSVType->getIntegerBitWidth() / 8;

        if (abi::RegisterState::shouldEmit(RSArg)) {
          NamedTypedRegister TR(RegisterID);
          TR.Type() = {
            Binary->getPrimitiveType(PrimitiveTypeKind::Generic, CSVSize), {}
          };
          ArgumentsInserter.insert(TR);
        }

        if (abi::RegisterState::shouldEmit(RSRV)) {
          NamedTypedRegister TR(RegisterID);
          TR.Type() = {
            Binary->getPrimitiveType(PrimitiveTypeKind::Generic, CSVSize), {}
          };
          ReturnValuesInserter.insert(TR);
        }
      }

      // Preserved registers
      const auto &ClobberedRegisters = Summary.ClobberedRegisters;
      auto PreservedRegisters = computePreservedRegisters(ClobberedRegisters);
      FunctionType.PreservedRegisters() = std::move(PreservedRegisters);

      // Final stack offset
      FunctionType.FinalStackOffset() = Summary.ElectedFSO.value_or(0);
    }

    Function.Prototype() = Binary->recordNewType(std::move(NewType));
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
  }

  for (auto &Function : Functions) {
    auto &Summary = Oracle.getLocalFunction(Function->Entry());
    efa::FunctionMetadata FM(Function->Entry(), Summary.CFG);
    FM.verify(*Binary, true);
  }

  revng_check(Binary->verify(true));
}

void DetectABI::propagatePrototypes() {
  for (model::Function &Function : Binary->Functions()) {
    propagatePrototypesInFunction(Function);
  }
}

void DetectABI::propagatePrototypesInFunction(model::Function &Function) {
  const MetaAddress &Entry = Function.Entry();

  revng_log(Log, "Trying to propagate prototypes for " << Entry.toString());
  LoggerIndent<> Indent(Log);

  FunctionSummary &Summary = Oracle.getLocalFunction(Entry);
  ABIAnalyses::ABIAnalysesResults &ABI = Summary.ABIResults;
  SortedVector<efa::BasicBlock> &CFG = Summary.CFG;
  std::set<llvm::GlobalVariable *> &WrittenRegisters = Summary.WrittenRegisters;

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
    model::TypePath Prototype = getPrototype(*Binary, Entry, Block, *Call);

    using abi::FunctionType::Layout;
    // Get layout of wrapped function
    Layout CalleeLayout = Layout::make(Prototype);

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
    // wrapped function prototype (CABIFunctionType or RawFunctionType)
    if (not WritesSP and not WritesCalleeArgs and not WritesCalleeReturnValues
        and WritesOnlyRegisters) {

      revng_log(Log,
                "Overwriting " << Entry.toString() << " prototype ("
                               << Function.Prototype().toString()
                               << ") with wrapped function's prototype: "
                               << Prototype.toString());
      Function.Prototype() = Prototype;

      if (Function.CustomName().empty() and Function.OriginalName().empty()) {
        if (not Call->DynamicFunction().empty()) {
          Function.OriginalName() = Call->DynamicFunction();
        } else if (Call->Destination().isValid()) {
          const model::Function &Callee = Binary->Functions()
                                            .at(Call->Destination()
                                                  .notInlinedAddress());
          if (not Callee.CustomName().empty())
            Function.OriginalName() = Callee.CustomName().str();
          else if (not Callee.OriginalName().empty())
            Function.OriginalName() = Callee.OriginalName();
        }
      }
    }
  }

  // TODO: should this be done at an higher abstraction level?
  model::promoteOriginalName(Binary);
}

UpcastablePointer<model::Type>
DetectABI::buildPrototypeForIndirectCall(const FunctionSummary &CallerSummary,
                                         const efa::BasicBlock &CallerBlock) {
  using namespace model;
  using RegisterState = abi::RegisterState::Values;

  auto NewType = makeType<RawFunctionType>();
  auto &CallType = *llvm::cast<RawFunctionType>(NewType.get());
  {
    auto ArgumentsInserter = CallType.Arguments().batch_insert();
    auto ReturnValuesInserter = CallType.ReturnValues().batch_insert();

    bool Found = false;
    for (const auto &[PC, CallSites] : CallerSummary.ABIResults.CallSites) {
      if (PC != CallerBlock.ID())
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
                                                       Binary->Architecture());
        if (RegisterID == Register::Invalid || CSV == GCBI.spReg())
          continue;

        auto *CSVType = CSV->getValueType();
        auto CSVSize = CSVType->getIntegerBitWidth() / 8;

        using namespace PrimitiveTypeKind;
        TypePath GenericType = Binary->getPrimitiveType(Generic, CSVSize);

        if (abi::RegisterState::shouldEmit(RSArg)) {
          NamedTypedRegister TR(RegisterID);
          TR.Type() = { GenericType, {} };
          ArgumentsInserter.insert(TR);
        }

        if (abi::RegisterState::shouldEmit(RSRV)) {
          NamedTypedRegister TR(RegisterID);
          TR.Type() = { GenericType, {} };
          ReturnValuesInserter.insert(TR);
        }
      }
    }
    revng_assert(Found);

    // Import FinalStackOffset and CalleeSavedRegisters from the default
    // prototype
    const FunctionSummary &DefaultSummary = Oracle.getDefault();
    const auto &Clobbered = DefaultSummary.ClobberedRegisters;
    CallType.PreservedRegisters() = computePreservedRegisters(Clobbered);

    CallType.FinalStackOffset() = DefaultSummary.ElectedFSO.value_or(0);
  }

  return NewType;
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
      if (ABIResults.CallSites[K].ArgumentsRegisters.contains(Reg))
        ABIResults.CallSites[K].ArgumentsRegisters[Reg] = RegisterState::No;

      if (ABIResults.CallSites[K].ReturnValuesRegisters.contains(Reg))
        ABIResults.CallSites[K].ReturnValuesRegisters[Reg] = RegisterState::No;
    }
  }
}

bool DetectABI::runAnalyses(MetaAddress EntryAddress,
                            OutlinedFunction &OutlinedFunction) {
  using namespace llvm;
  using llvm::BasicBlock;
  using namespace ABIAnalyses;

  IRBuilder<> Builder(M.getContext());
  ABIAnalysesResults ABIResults;

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
  FunctionSummary &Summary = Oracle.getLocalFunction(EntryAddress);
  auto CalleeSavedRegs = computePreservedCSVs(Summary.ClobberedRegisters);
  auto ActualCalleeSavedRegs = intersect(CalleeSavedRegs, WrittenRegisters);

  // Union between effective callee-saved registers and SP
  ActualCalleeSavedRegs.insert(GCBI.spReg());

  // Refine ABI analyses results by suppressing callee-saved and stack
  // pointer registers.
  suppressCSAndSPRegisters(ABIResults, ActualCalleeSavedRegs);

  // Merge return values registers
  ABIAnalyses::finalizeReturnValues(ABIResults);

  // Commit ABI analysis results to the oracle
  auto Old = Summary.ABIResults;
  ABIResults.normalize();
  Summary.ABIResults.combine(ABIResults);
  Summary.WrittenRegisters = WrittenRegisters;

  bool Changed = Old != Summary.ABIResults;

  for (auto &[BlockID, CallSite] : ABIResults.CallSites) {
    if (BlockID.isInlined())
      continue;

    auto Set = [&Changed](abi::RegisterState::Values &Value) {
      const auto Yes = abi::RegisterState::Yes;
      if (Value != Yes) {
        Value = Yes;
        Changed = true;
      }
    };

    // TODO: eventually we'll want to add arguments/return values to dynamic
    //       functions too
    MetaAddress Callee = CallSite.CalleeAddress;
    if (Callee.isValid()) {
      auto &CalleeSummary = Oracle.getLocalFunction(Callee);

      for (auto &[CSV, Value] : CallSite.ArgumentsRegisters)
        Set(CalleeSummary.ABIResults.ArgumentsRegisters[CSV]);

      for (auto &[CSV, Value] : CallSite.ReturnValuesRegisters)
        Set(CalleeSummary.ABIResults.FinalReturnValuesRegisters[CSV]);
    }
  }

  return Changed;
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

char DetectABIPass::ID = 0;

using ABIDetectionPass = RegisterPass<DetectABIPass>;
static ABIDetectionPass X("detect-abi", "ABI Detection Pass", true, false);

} // namespace efa
