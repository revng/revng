/// \file IsolateFunctions.cpp
/// Implements the IsolateFunctions pass which applies function isolation using
/// the information provided by EarlyFunctionAnalysis.

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/ADT/DepthFirstIterator.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/CFG.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DIBuilder.h"
#include "llvm/IR/DebugInfoMetadata.h"
#include "llvm/IR/GlobalValue.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Support/raw_os_ostream.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/Transforms/Utils/CodeExtractor.h"
#include "llvm/Transforms/Utils/Local.h"

#include "revng/ADT/KeyedObjectContainer.h"
#include "revng/ADT/Queue.h"
#include "revng/ADT/ZipMapIterator.h"
#include "revng/BasicAnalyses/GeneratedCodeBasicInfo.h"
#include "revng/EarlyFunctionAnalysis/AnalyzeRegisterUsage.h"
#include "revng/EarlyFunctionAnalysis/BasicBlock.h"
#include "revng/EarlyFunctionAnalysis/CallHandler.h"
#include "revng/EarlyFunctionAnalysis/ControlFlowGraph.h"
#include "revng/EarlyFunctionAnalysis/ControlFlowGraphCache.h"
#include "revng/EarlyFunctionAnalysis/FunctionEdge.h"
#include "revng/EarlyFunctionAnalysis/FunctionEdgeBase.h"
#include "revng/EarlyFunctionAnalysis/FunctionSummaryOracle.h"
#include "revng/EarlyFunctionAnalysis/Generated/ForwardDecls.h"
#include "revng/EarlyFunctionAnalysis/Outliner.h"
#include "revng/FunctionIsolation/IsolateFunctions.h"
#include "revng/Model/Binary.h"
#include "revng/Model/NameBuilder.h"
#include "revng/Pipeline/AllRegistries.h"
#include "revng/Pipeline/Contract.h"
#include "revng/Pipeline/ExecutionContext.h"
#include "revng/Pipes/Kinds.h"
#include "revng/Pipes/ModelGlobal.h"
#include "revng/Pipes/RootKind.h"
#include "revng/Pipes/StringMap.h"
#include "revng/Pipes/TaggedFunctionKind.h"
#include "revng/Support/Debug.h"
#include "revng/Support/FunctionTags.h"
#include "revng/Support/IRHelpers.h"
#include "revng/Support/MetaAddress.h"

using namespace llvm;

class IsolateFunctionsImpl;

static Logger<> TheLogger("isolation");

// Define an alias for the data structure that will contain the LLVM functions
using FunctionsMap = std::map<MDString *, Function *>;
using ValueToValueMap = DenseMap<const Value *, Value *>;

using IF = IsolateFunctions;
using IFI = IsolateFunctionsImpl;

char IF::ID = 0;
static RegisterPass<IF> X("isolate", "Isolate Functions Pass", true, true);

struct IsolatePipe {
  static constexpr auto Name = "isolate";

  std::vector<pipeline::ContractGroup> getContract() const {
    using namespace revng;
    using namespace pipeline;
    return { ContractGroup({ Contract(kinds::Root,
                                      1,
                                      kinds::Isolated,
                                      1,
                                      InputPreservation::Preserve),
                             Contract(kinds::CFG,
                                      0,
                                      kinds::Isolated,
                                      1,
                                      InputPreservation::Preserve) }) };
  }

  void run(pipeline::ExecutionContext &EC,
           const revng::pipes::CFGMap &CFGMap,
           pipeline::LLVMContainer &ModuleContainer) {
    using namespace revng;
    llvm::legacy::PassManager Manager;
    Manager.add(new pipeline::LoadExecutionContextPass(&EC,
                                                       ModuleContainer.name()));
    Manager
      .add(new LoadModelWrapperPass(ModelWrapper(getModelFromContext(EC))));
    Manager.add(new ControlFlowGraphCachePass(CFGMap));
    Manager.add(new IsolateFunctions());
    Manager.run(ModuleContainer.getModule());
  }
};

static pipeline::RegisterPipe<IsolatePipe> Y;

struct Boundary {
  BasicBlock *Block = nullptr;
  BasicBlock *CalleeBlock = nullptr;
  BasicBlock *ReturnBlock = nullptr;

  bool isCall() const { return ReturnBlock != nullptr; }

  void dump() const debug_function { dump(dbg); }

  template<typename O>
  void dump(O &Output) const {
    Output << "Block: " << getName(Block) << "\n";
    Output << "CalleeBlock: " << getName(CalleeBlock) << "\n";
    Output << "ReturnBlock: " << getName(ReturnBlock) << "\n";
  }
};

class FunctionBlocks {
private:
  enum FixedBlocks {
    DummyEntryBlock,
    ReturnBlock,
    UnexpectedPCBlock,
    FixedBlocksCount
  };

public:
  SmallVector<BasicBlock *, 16> Blocks;

public:
  BasicBlock *&dummyEntryBlock() { return Blocks[DummyEntryBlock]; }
  BasicBlock *&returnBlock() { return Blocks[ReturnBlock]; }
  BasicBlock *&unexpectedPCBlock() { return Blocks[UnexpectedPCBlock]; }

public:
  FunctionBlocks() : Blocks(FixedBlocksCount) {}

  auto begin() { return Blocks.begin(); }
  auto end() { return Blocks.end(); }

  void push_back(BasicBlock *BB) { Blocks.push_back(BB); }

  bool contains(BasicBlock *BB) const {
    return llvm::find(Blocks, BB) != Blocks.end();
  }
};

class IsolateFunctionsImpl {
private:
  using SuccessorsContainer = std::map<const efa::FunctionEdgeBase *, int>;

private:
  Function *RootFunction = nullptr;
  Module *TheModule = nullptr;
  LLVMContext &Context;
  GeneratedCodeBasicInfo &GCBI;
  const model::Binary &Binary;
  model::NameBuilder NameBuilder;
  Function *AbortFunction = nullptr;
  Function *UnreachableFunction = nullptr;
  Function *FunctionDispatcher = nullptr;
  std::map<MetaAddress, Function *> IsolatedFunctionsMap;
  std::map<StringRef, Function *> DynamicFunctionsMap;

  ControlFlowGraphCache *Cache = nullptr;
  FunctionType *IsolatedFunctionType = nullptr;
  pipeline::LoadExecutionContextPass &LECP;

public:
  IsolateFunctionsImpl(Function *RootFunction,
                       GeneratedCodeBasicInfo &GCBI,
                       const model::Binary &Binary,
                       ControlFlowGraphCache &Cache,
                       pipeline::LoadExecutionContextPass &LECP) :
    RootFunction(RootFunction),
    TheModule(RootFunction->getParent()),
    Context(TheModule->getContext()),
    GCBI(GCBI),
    Binary(Binary),
    NameBuilder(Binary),
    Cache(&Cache),
    LECP(LECP) {
    IsolatedFunctionType = createFunctionType<void>(Context);
  }

public:
  Function *getLocalFunction(const MetaAddress &Entry) {
    auto Name = NameBuilder.llvmName(Binary.Functions().at(Entry));
    if (auto *F = TheModule->getFunction(Name))
      return F;

    auto *F = Function::Create(IsolatedFunctionType,
                               GlobalValue::ExternalLinkage,
                               Name,
                               TheModule);
    FunctionTags::Isolated.addTo(F);
    setMetaAddressMetadata(F, FunctionEntryMDName, Entry);
    return F;
  }

  Function *getDynamicFunction(llvm::StringRef SymbolName) const {
    return DynamicFunctionsMap.at(SymbolName);
  }

  Function *dispatcher() const { return FunctionDispatcher; }
  auto &gcbi() const { return GCBI; }

public:
  void run();

  void emitAbort(IRBuilder<> &Builder,
                 const Twine &Reason,
                 const DebugLoc &DbgLocation) {
    emitCall(Builder,
             AbortFunction,
             Reason,
             DbgLocation,
             GCBI.programCounterHandler());
  }

  void
  emitAbort(BasicBlock *BB, const Twine &Reason, const DebugLoc &DbgLocation) {
    IRBuilder<> Builder(BB);
    emitAbort(Builder, Reason, DbgLocation);
  }

  void emitUnreachable(IRBuilder<> &Builder,
                       const Twine &Reason,
                       const DebugLoc &DbgLocation) {

    emitCall(Builder,
             UnreachableFunction,
             Reason,
             DbgLocation,
             GCBI.programCounterHandler());
  }

  void emitUnreachable(BasicBlock *BB,
                       const Twine &Reason,
                       const DebugLoc &DbgLocation) {
    IRBuilder<> Builder(BB);
    emitUnreachable(Builder, Reason, DbgLocation);
  }

private:
  /// Populate the function_dispatcher, needed to handle the indirect calls
  void populateFunctionDispatcher();

  void handleUnexpectedPCCloned(efa::OutlinedFunction &Outlined);
  void handleAnyPCJumps(efa::OutlinedFunction &Outlined,
                        const efa::ControlFlowGraph &FM);
};

void IFI::populateFunctionDispatcher() {

  BasicBlock *Dispatcher = BasicBlock::Create(Context,
                                              "function_dispatcher",
                                              FunctionDispatcher,
                                              nullptr);

  BasicBlock *Unexpected = BasicBlock::Create(Context,
                                              "unexpectedpc",
                                              FunctionDispatcher,
                                              nullptr);
  const DebugLoc &Dbg = GCBI.unexpectedPC()->getTerminator()->getDebugLoc();
  emitUnreachable(Unexpected, "An unexpected function has been called", Dbg);
  setBlockType(Unexpected->getTerminator(), BlockType::UnexpectedPCBlock);

  IRBuilder<> Builder(Context);

  // Create all the entries of the dispatcher
  ProgramCounterHandler::DispatcherTargets Targets;
  for (auto &[Address, F] : IsolatedFunctionsMap) {
    BasicBlock *Trampoline = BasicBlock::Create(Context,
                                                F->getName() + "_trampoline",
                                                FunctionDispatcher,
                                                nullptr);
    Targets.emplace_back(Address, Trampoline);

    Builder.SetInsertPoint(Trampoline);
    Builder.CreateCall(F);
    Builder.CreateRetVoid();
  }

  // Create switch
  Builder.SetInsertPoint(Dispatcher);
  GCBI.programCounterHandler()->buildDispatcher(Targets,
                                                Builder,
                                                Unexpected,
                                                {});
}

template<typename T, typename F>
static bool
allOrNone(const T &Range, const F &Predicate, bool Default = false) {
  auto Start = Range.begin();
  auto End = Range.end();

  if (Start == End)
    return Default;

  bool First = Predicate(*Start);
  ++Start;
  for (const auto &E : make_range(Start, End))
    revng_assert(First == Predicate(E));

  return First;
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

struct SetAtMostOnce {
private:
  bool State = false;

public:
  bool get() const { return State; }
  void set() {
    revng_assert(not State);
    State = true;
  }

  void setIf(bool Condition) {
    if (Condition)
      set();
  }

  operator bool() const { return State; }
};

template<typename LeftMap, typename RightMap>
void printAddressListComparison(const LeftMap &ExpectedAddresses,
                                const RightMap &ActualAddresses) {
  // Compare expected and actual
  if (TheLogger.isEnabled()) {
    for (auto &&[ExpectedAddress, ActualAddress] :
         zipmap_range(ExpectedAddresses, ActualAddresses)) {
      if (ExpectedAddress == nullptr) {
        TheLogger << "Warning: ";
        ActualAddress->dump(TheLogger);
        TheLogger << " detected as a jump target, but the model does not list "
                     "it"
                  << DoLog;
      } else if (ActualAddress == nullptr) {
        TheLogger << "Warning: ";
        ExpectedAddress->dump(TheLogger);
        TheLogger << " not detected as a jump target, but the model lists it"
                  << DoLog;
      }
    }
  }
}

class CallIsolatedFunction : public efa::CallHandler {
private:
  IsolateFunctionsImpl &IFI;
  const efa::ControlFlowGraph &FM;

public:
  CallIsolatedFunction(IsolateFunctionsImpl &IFI,
                       const efa::ControlFlowGraph &FM) :
    IFI(IFI), FM(FM) {}

public:
  void handleCall(MetaAddress CallerBlock,
                  llvm::IRBuilder<> &Builder,
                  MetaAddress Callee,
                  const efa::CSVSet &ClobberedRegisters,
                  const std::optional<int64_t> &MaybeFSO,
                  bool IsNoReturn,
                  bool IsTailCall,
                  llvm::Value *SymbolNamePointer) final {
    revng_assert(MaybeFSO == std::nullopt,
                 "FSO is expensive to compute for CFT but is not used, "
                 "is there maybe a way to avoid it?");
    handleCall(Builder, Callee, SymbolNamePointer);
  }

  void handlePostNoReturn(llvm::IRBuilder<> &Builder) final {
    // TODO: can we do better than DebugLoc()?
    IFI.emitUnreachable(Builder,
                        "We return from a noreturn function call",
                        DebugLoc());
  }

  void handleIndirectJump(llvm::IRBuilder<> &Builder,
                          MetaAddress Block,
                          const efa::CSVSet &ClobberedRegisters,
                          llvm::Value *SymbolNamePointer) final {
    revng_assert(SymbolNamePointer != nullptr);
    if (not isa<ConstantPointerNull>(SymbolNamePointer))
      handleCall(Builder, MetaAddress::invalid(), SymbolNamePointer);
  }

private:
  void handleCall(llvm::IRBuilder<> &Builder,
                  MetaAddress Callee,
                  llvm::Value *SymbolNamePointer) {
    // Identify caller block
    const auto *Caller = FM.findBlock(IFI.gcbi(), Builder.GetInsertBlock());

    // Identify call edge
    auto IsCallEdge = [](const UpcastablePointer<efa::FunctionEdgeBase> &E) {
      return isa<efa::CallEdge>(E.get());
    };
    auto ZeroOrOneCallEdge = [](const auto &Range,
                                const auto &Callable) -> const efa::CallEdge * {
      auto *Result = zeroOrOne(Range, Callable);
      if (Result == nullptr)
        return nullptr;
      else
        return dyn_cast<efa::CallEdge>(Result->get());
    };
    const auto *CallEdge = ZeroOrOneCallEdge(Caller->Successors(), IsCallEdge);

    if (CallEdge == nullptr) {
      // There's no CallEdge, this is likely a LongJmp
      return;
    }

    StringRef SymbolName = extractFromConstantStringPtr(SymbolNamePointer);
    revng_assert(SymbolName == CallEdge->DynamicFunction());

    if (Callee != CallEdge->Destination().notInlinedAddress()) {
      revng_assert(not CallEdge->Destination().notInlinedAddress().isValid());

      // The callee in the IR is different from the one we get from the CFG.
      // This likely means that the called address is not a function.
      // For now, we represent this as an indirect function call.
      Callee = MetaAddress::invalid();
    }

    // Identify callee
    Function *CalledFunction = nullptr;
    if (Callee.isValid())
      CalledFunction = IFI.getLocalFunction(Callee);
    else if (not SymbolName.empty())
      CalledFunction = IFI.getDynamicFunction(SymbolName);
    else
      CalledFunction = IFI.dispatcher();

    //
    // Create the call
    //
    BasicBlock::iterator InsertPoint = Builder.GetInsertPoint();
    revng_assert(not Builder.GetInsertBlock()->empty());
    bool AtEnd = InsertPoint == Builder.GetInsertBlock()->end();
    Instruction *Old = AtEnd ? &*Builder.GetInsertBlock()->rbegin() :
                               &*InsertPoint;
    auto *NewCall = Builder.CreateCall(CalledFunction);
    NewCall->addFnAttr(Attribute::NoMerge);
    NewCall->setDebugLoc(Old->getDebugLoc());
  }
};

template<typename R>
inline auto toVector(R &&Range) {
  using ResultType = decltype(*Range.begin());
  SmallVector<ResultType> Result;
  for (auto Element : Range)
    Result.push_back(Element);
  return Result;
}

using FSOracle = efa::FunctionSummaryOracle;

class FunctionOutliner {
private:
  efa::FunctionSummaryOracle Oracle;
  efa::Outliner Outliner;

public:
  FunctionOutliner(llvm::Module &M,
                   const model::Binary &Binary,
                   GeneratedCodeBasicInfo &GCBI) :
    Oracle(FSOracle::importWithoutPrototypes(M, GCBI, Binary)),
    Outliner(M, GCBI, Oracle) {}

public:
  efa::OutlinedFunction outline(MetaAddress Entry,
                                efa::CallHandler *TheCallHandler) {
    return Outliner.outline(Entry, TheCallHandler);
  }
};

void IsolateFunctionsImpl::run() {
  Task T(6, "IsolateFunctions");

  AbortFunction = TheModule->getFunction("_abort");
  revng_assert(AbortFunction != nullptr);

  UnreachableFunction = TheModule->getFunction("_unreachable");
  revng_assert(UnreachableFunction != nullptr);
  FunctionTags::Exceptional.addTo(UnreachableFunction);

  FunctionDispatcher = Function::Create(createFunctionType<void>(Context),
                                        GlobalValue::ExternalLinkage,
                                        "function_dispatcher",
                                        TheModule);
  FunctionTags::FunctionDispatcher.addTo(FunctionDispatcher);

  //
  // Create the dynamic functions
  //
  // TODO: we can (and should) push processing of dynamic functions into the
  //       loop emitting individual local functions, and make it lazy
  T.advance("Create dynamic functions", true);
  Task DynamicFunctionsTask(Binary.ImportedDynamicFunctions().size(),
                            "Dynamic functions creation");
  for (const model::DynamicFunction &Function :
       Binary.ImportedDynamicFunctions()) {
    StringRef Name = Function.OriginalName();
    DynamicFunctionsTask.advance(Name, true);
    auto *NewFunction = Function::Create(IsolatedFunctionType,
                                         GlobalValue::ExternalLinkage,
                                         "dynamic_" + Function.OriginalName(),
                                         TheModule);
    FunctionTags::DynamicFunction.addTo(NewFunction);
    NewFunction->addFnAttr(Attribute::NoMerge);

    auto *EntryBB = BasicBlock::Create(Context, "", NewFunction);
    emitAbort(EntryBB, Twine("Dynamic call ") + Name, DebugLoc());

    // TODO: implement more efficient version.
    // if (setjmp(...) == 0) {
    //   // First return
    //   serialize_cpu_state();
    //   dynamic_function();
    //   // If we get here, it means that the external function return properly
    //   deserialize_cpu_state();
    //   simulate_ret();
    //   // If the caller tail-called us, it must return immediately, without
    //   // checking if the pc is the fallthrough of the call (which was not a
    //   // call!)
    // } else {
    //   // If we get here, it means that the external function either invoked a
    //   // callback or something else weird i going on.
    //   deserialize_cpu_state();
    //   throw_exception();
    // }

    DynamicFunctionsMap[Name] = NewFunction;
  }

  using namespace efa;
  using llvm::BasicBlock;

  T.advance("Create dynamic functions", true);

  // Obtain the set of requested targets
  pipeline::ExecutionContext &Context = *LECP.get();
  const pipeline::TargetsList &RequestedTargets = LECP.getRequestedTargets();

  Task IsolateTask(RequestedTargets.size(), "Isolating functions");
  for (const pipeline::Target &Target : RequestedTargets) {
    IsolateTask.advance(Target.toString(), true);
    Context.getContext().pushReadFields();

    auto Entry = MetaAddress::fromString(Target.getPathComponents()[0]);
    const efa::ControlFlowGraph &FM = Cache->getControlFlowGraph(Entry);

    // Get or create the llvm::Function
    Function *F = getLocalFunction(Entry);

    // Decorate the function as appropriate
    F->addFnAttr(Attribute::NullPointerIsValid);
    F->addFnAttr(Attribute::NoMerge);
    IsolatedFunctionsMap[Entry] = F;
    revng_assert(F != nullptr);

    // Outline the function (later on we'll steal its body and move it into F)
    CallIsolatedFunction CallHandler(*this, FM);
    FunctionOutliner Outliner(*TheModule, Binary, GCBI);
    OutlinedFunction Outlined = Outliner.outline(Entry, &CallHandler);

    handleUnexpectedPCCloned(Outlined);

    handleAnyPCJumps(Outlined, FM);

    if (Outlined.Function)
      for (BasicBlock &BB : *Outlined.Function)
        revng_assert(BB.getTerminator() != nullptr);

    // Steal the function body and let the outlined function be destroyed
    moveBlocksInto(*Outlined.Function, *F);

    // Commit the produced target
    Context.commit(Target, LECP.getContainerName());

    Context.getContext().popReadFields();
  }

  T.advance("Verify module", true);
  revng::verify(TheModule);

  // Create the functions and basic blocks needed for the correct execution of
  // the exception handling mechanism

  // Populate the function_dispatcher
  T.advance("Populate function_dispatcher", true);
  populateFunctionDispatcher();

  // Cleanup root
  T.advance("Cleanup", true);
  EliminateUnreachableBlocks(*RootFunction, nullptr, false);

  // Before emitting it in output, verify the module
  T.advance("Verify module", true);
  revng::verify(TheModule);
}

void IsolateFunctionsImpl::handleUnexpectedPCCloned(efa::OutlinedFunction
                                                      &Outlined) {
  if (BasicBlock *UnexpectedPC = Outlined.UnexpectedPCCloned) {
    for (auto It = UnexpectedPC->begin(); It != UnexpectedPC->end();
         It = UnexpectedPC->begin())
      It->eraseFromParent();
    revng_assert(UnexpectedPC->empty());
    const DebugLoc &Dbg = GCBI.unexpectedPC()->getTerminator()->getDebugLoc();
    emitUnreachable(UnexpectedPC, "unexpectedPC", Dbg);
  }
}

void IsolateFunctionsImpl::handleAnyPCJumps(efa::OutlinedFunction &Outlined,
                                            const efa::ControlFlowGraph &FM) {
  if (BasicBlock *AnyPC = Outlined.AnyPCCloned) {
    for (BasicBlock *AnyPCPredecessor : toVector(predecessors(AnyPC))) {
      // First of all, identify the basic block
      const efa::BasicBlock *Block = FM.findBlock(GCBI, AnyPCPredecessor);

      Instruction *T = AnyPCPredecessor->getTerminator();
      revng_assert(not cast<BranchInst>(T)->isConditional());
      T->eraseFromParent();
      IRBuilder<> Builder(AnyPCPredecessor);

      // Get the only outgoing edge jumping to anypc
      if (Block == nullptr) {
        emitAbort(Builder, "Unexpected jump", DebugLoc());
        continue;
      }

      bool AtLeastAMatch = false;
      for (auto &Edge : Block->Successors()) {
        if (Edge->Type() == efa::FunctionEdgeType::DirectBranch)
          continue;

        revng_assert(not AtLeastAMatch);
        AtLeastAMatch = true;

        switch (Edge->Type()) {
        case efa::FunctionEdgeType::Return:
          Builder.CreateRetVoid();
          break;
        case efa::FunctionEdgeType::BrokenReturn:
          // TODO: can we do better than DebugLoc()?
          emitAbort(Builder, "A broken return was taken", DebugLoc());
          break;
        case efa::FunctionEdgeType::LongJmp:
          emitAbort(Builder, "A longjmp was taken", DebugLoc());
          break;
        case efa::FunctionEdgeType::Killer:
          emitAbort(Builder, "A killer block has been reached", DebugLoc());
          revng_abort();
          break;
        case efa::FunctionEdgeType::Unreachable:
          emitAbort(Builder,
                    "An unreachable instruction has been "
                    "reached",
                    DebugLoc());
          break;
        case efa::FunctionEdgeType::FunctionCall: {
          auto *Call = cast<efa::CallEdge>(Edge.get());
          revng_assert(Call->IsTailCall());
          Builder.CreateRetVoid();
        } break;
        case efa::FunctionEdgeType::Invalid:
        case efa::FunctionEdgeType::DirectBranch:
        case efa::FunctionEdgeType::Count:
          revng_abort();
          break;
        }
      }

      if (not AtLeastAMatch) {
        emitAbort(Builder, "Unexpected jump", DebugLoc());
        continue;
      }
    }

    eraseFromParent(AnyPC);
  }
}

bool IF::runOnModule(Module &TheModule) {
  if (not TheModule.getFunction("root")
      or TheModule.getFunction("root")->isDeclaration())
    return false;

  //  Retrieve analyses
  auto &GCBI = getAnalysis<GeneratedCodeBasicInfoWrapperPass>().getGCBI();
  const auto &ModelWrapper = getAnalysis<LoadModelWrapperPass>().get();
  const model::Binary &Binary = *ModelWrapper.getReadOnlyModel();

  // Create an object of type IsolateFunctionsImpl and run the pass
  IFI Impl(TheModule.getFunction("root"),
           GCBI,
           Binary,
           getAnalysis<ControlFlowGraphCachePass>().get(),
           getAnalysis<pipeline::LoadExecutionContextPass>());
  Impl.run();

  return false;
}

void IsolateFunctions::getAnalysisUsage(llvm::AnalysisUsage &AU) const {
  AU.setPreservesAll();
  AU.addRequired<GeneratedCodeBasicInfoWrapperPass>();
  AU.addRequired<LoadModelWrapperPass>();
  AU.addRequired<ControlFlowGraphCachePass>();
  AU.addRequired<pipeline::LoadExecutionContextPass>();
}
