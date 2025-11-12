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
#include "llvm/IR/Instructions.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Support/raw_os_ostream.h"
#include "llvm/Transforms/IPO.h"
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
#include "revng/EarlyFunctionAnalysis/Outliner.h"
#include "revng/FunctionIsolation/IsolateFunctions.h"
#include "revng/Model/Binary.h"
#include "revng/Model/FunctionTags.h"
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
#include "revng/Support/IRHelpers.h"
#include "revng/Support/MetaAddress.h"

// This name is not present after `enforce-abi`.
RegisterIRHelper FDispatcher("function_dispatcher");

using namespace llvm;

class IsolateFunctionsImpl;

static Logger TheLogger("isolation");

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
                                      2,
                                      InputPreservation::Preserve),
                             Contract(kinds::CFG,
                                      0,
                                      kinds::Isolated,
                                      2,
                                      InputPreservation::Preserve) }) };
  }

private:
  static void cleanTheModuleUp(llvm::Module &Module) {
    // Is there something in LLVM that does this already?

    // WARNING: this removes way too much, breaking some assumptions
    // we have about the module.
    // TODO: investigate the proper way to do this.

    // llvm::legacy::PassManager PM;
    // PM.add(llvm::createStripDeadPrototypesPass());
    // PM.add(llvm::createGlobalDCEPass());
    // PM.add(llvm::createDeadArgEliminationPass());
    // PM.add(llvm::createStripDeadDebugInfoPass());
    // for (int i = 0; i < 3; ++i) {
    //   PM.add(llvm::createGlobalDCEPass());
    //   PM.add(llvm::createStripDeadPrototypesPass());
    // }

    // PM.run(Module);
  }

public:
  void run(pipeline::ExecutionContext &EC,
           const revng::pipes::CFGMap &CFGMap,
           pipeline::LLVMContainer &RootContainer,
           pipeline::LLVMContainer &OutputContainer) {
    // Clone the container
    OutputContainer.cloneFrom(RootContainer);

    // Do the isolation
    using namespace revng;
    llvm::legacy::PassManager Manager;
    Manager.add(new pipeline::LoadExecutionContextPass(&EC,
                                                       OutputContainer.name()));
    Manager
      .add(new LoadModelWrapperPass(ModelWrapper(getModelFromContext(EC))));
    Manager.add(new ControlFlowGraphCachePass(CFGMap));
    Manager.add(new IsolateFunctions());
    Manager.run(OutputContainer.getModule());

    // Remove "root" from the output container.
    namespace FT = FunctionTags;
    for (Function &F : FT::Root.functions(&OutputContainer.getModule()))
      F.deleteBody();

    // Finally, do some basic cleanup, removing unused stuff.
    cleanTheModuleUp(RootContainer.getModule());
    cleanTheModuleUp(OutputContainer.getModule());
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
  using CFG = efa::ControlFlowGraph;
  using CFGGetterType = std::function<const CFG &(const MetaAddress &)>;

private:
  Function *RootFunction = nullptr;
  Module *TheModule = nullptr;
  LLVMContext &Context;
  GeneratedCodeBasicInfo &GCBI;
  const model::Binary &Binary;
  model::CNameBuilder NameBuilder;
  Function *FunctionDispatcher = nullptr;
  std::map<MetaAddress, Function *> IsolatedFunctionsMap;
  std::map<StringRef, Function *> DynamicFunctionsMap;

  CFGGetterType CFGGetter;
  FunctionType *IsolatedFunctionType = nullptr;

public:
  IsolateFunctionsImpl(Function *RootFunction,
                       GeneratedCodeBasicInfo &GCBI,
                       const model::Binary &Binary,
                       CFGGetterType CFGGetter) :
    RootFunction(RootFunction),
    TheModule(RootFunction->getParent()),
    Context(TheModule->getContext()),
    GCBI(GCBI),
    Binary(Binary),
    NameBuilder(Binary),
    CFGGetter(CFGGetter) {
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
  void prologue();
  llvm::Function *runOnFunction(const MetaAddress &Address);
  void epilogue();

  void emitAbort(revng::IRBuilder &Builder,
                 const Twine &Reason,
                 const DebugLoc &DbgLocation) {
    ::emitAbort(Builder, Reason, DbgLocation, GCBI.programCounterHandler());
  }

  void
  emitAbort(BasicBlock *BB, const Twine &Reason, const DebugLoc &DbgLocation) {
    revng::NonDebugInfoCheckingIRBuilder Builder(BB);
    emitAbort(Builder, Reason, DbgLocation);
  }

  void emitUnreachable(revng::IRBuilder &Builder,
                       const Twine &Reason,
                       const DebugLoc &DbgLocation) {
    // Emitting any long-lasting messages here prevents switch detection,
    // so use a simple `unreachable`.
    Builder.CreateUnreachable();
  }

  void emitUnreachable(BasicBlock *BB,
                       const Twine &Reason,
                       const DebugLoc &DbgLocation) {
    revng::NonDebugInfoCheckingIRBuilder Builder(BB);
    emitUnreachable(Builder, Reason, DbgLocation);
  }

private:
  void handleUnexpectedPCCloned(efa::OutlinedFunction &Outlined);
  void handleAnyPCJumps(efa::OutlinedFunction &Outlined,
                        const efa::ControlFlowGraph &FM);
};

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
                  revng::IRBuilder &Builder,
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

  void handlePostNoReturn(revng::IRBuilder &Builder,
                          const llvm::DebugLoc &DbgLocation) final {
    IFI.emitUnreachable(Builder,
                        "We return from a noreturn function call",
                        DbgLocation);
  }

  void handleIndirectJump(revng::IRBuilder &Builder,
                          MetaAddress Block,
                          const efa::CSVSet &ClobberedRegisters,
                          llvm::Value *SymbolNamePointer) final {
    revng_assert(SymbolNamePointer != nullptr);
    if (not isa<ConstantPointerNull>(SymbolNamePointer))
      handleCall(Builder, MetaAddress::invalid(), SymbolNamePointer);
  }

private:
  void handleCall(revng::IRBuilder &Builder,
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

void IsolateFunctionsImpl::prologue() {
  auto SimpleFunctionType = createFunctionType<void>(Context);
  FunctionDispatcher = createIRHelper("function_dispatcher",
                                      *TheModule,
                                      SimpleFunctionType,
                                      GlobalValue::ExternalLinkage);
  FunctionTags::FunctionDispatcher.addTo(FunctionDispatcher);

  //
  // Create the dynamic functions
  //
  // TODO: we can (and should) push processing of dynamic functions into the
  //       loop emitting individual local functions, and make it lazy
  Task DynamicFunctionsTask(Binary.ImportedDynamicFunctions().size(),
                            "Dynamic functions creation");
  for (const model::DynamicFunction &Function :
       Binary.ImportedDynamicFunctions()) {
    StringRef Name = Function.Name();
    DynamicFunctionsTask.advance(Name, true);
    auto *NewFunction = Function::Create(IsolatedFunctionType,
                                         GlobalValue::ExternalLinkage,
                                         "dynamic_" + Function.Name(),
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
}

llvm::Function *IsolateFunctionsImpl::runOnFunction(const MetaAddress &Entry) {

  revng_assert(Entry.isValid());
  const efa::ControlFlowGraph &FM = CFGGetter(Entry);

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
  efa::OutlinedFunction Outlined = Outliner.outline(Entry, &CallHandler);

  handleUnexpectedPCCloned(Outlined);

  handleAnyPCJumps(Outlined, FM);

  if (Outlined.Function)
    for (BasicBlock &BB : *Outlined.Function)
      revng_assert(BB.getTerminator() != nullptr);

  // Steal the function body and let the outlined function be destroyed
  moveBlocksInto(*Outlined.Function, *F);

  return F;
}

void IsolateFunctionsImpl::epilogue() {
  llvm::Task T(3, "Isolate: epilogue");
  T.advance("Verify module", true);
  revng::verify(TheModule);

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
      const efa::BasicBlock *JumpBlock = FM.findBlock(GCBI, AnyPCPredecessor);

      Instruction *T = AnyPCPredecessor->getTerminator();
      revng_assert(not cast<BranchInst>(T)->isConditional());
      T->eraseFromParent();

      // TODO: the checks should be enabled conditionally based on the user.
      revng::NonDebugInfoCheckingIRBuilder Builder(AnyPCPredecessor);

      // Get the only outgoing edge jumping to anypc
      if (JumpBlock == nullptr) {
        emitAbort(Builder, "Unexpected jump", DebugLoc());
        continue;
      }

      bool AtLeastAMatch = false;
      for (auto &Edge : JumpBlock->Successors()) {
        auto EdgeType = Edge->Type();
        if (EdgeType == efa::FunctionEdgeType::DirectBranch
            or EdgeType == efa::FunctionEdgeType::Unexpected) {
          continue;
        }

        switch (EdgeType) {
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
        case efa::FunctionEdgeType::Unexpected:
        case efa::FunctionEdgeType::Count:
          revng_abort();
          break;
        }

        revng_assert(not AtLeastAMatch);
        AtLeastAMatch = true;
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

  auto &LECP = getAnalysis<pipeline::LoadExecutionContextPass>();
  pipeline::ExecutionContext &Context = *LECP.get();
  const pipeline::TargetsList &RequestedTargets = LECP.getRequestedTargets();

  auto &CFGC = getAnalysis<ControlFlowGraphCachePass>().get();
  auto CFGGetter =
    [&CFGC](const MetaAddress &Address) -> const efa::ControlFlowGraph & {
    return CFGC.getControlFlowGraph(Address);
  };

  llvm::Task MainTask(3, "Isolate functions");
  MainTask.advance("Isolate: prologue");

  // Create an object of type IsolateFunctionsImpl and run the pass
  IFI Impl(TheModule.getFunction("root"), GCBI, Binary, CFGGetter);

  Impl.prologue();

  MainTask.advance("Isolate: run on functions");
  Task IsolateTask(RequestedTargets.size(), "Isolating functions");
  for (const pipeline::Target &Target : RequestedTargets) {
    Context.getContext().pushReadFields();

    auto Entry = MetaAddress::fromString(Target.getPathComponents()[0]);
    IsolateTask.advance(Entry.toString(), true);
    Impl.runOnFunction(Entry);

    // Commit the produced target
    Context.commit(Target, LECP.getContainerName());
    Context.getContext().popReadFields();
  }

  MainTask.advance("Isolate: epilogue");
  Impl.epilogue();

  return false;
}

namespace revng::pypeline::piperuns {

Isolate::Isolate(const class Model &Model,
                 llvm::StringRef Config,
                 llvm::StringRef DynamicConfig,
                 const CFGMap &CFG,
                 LLVMRootContainer &Root,
                 LLVMFunctionContainer &Output) :
  Root(Root), Output(Output), GCBI(*Model.get().get()) {
  llvm::Module &Module = Root.getModule();
  GCBI.run(Module);

  auto CFGGetter =
    [&CFG](const MetaAddress &Address) -> const efa::ControlFlowGraph & {
    return *CFG.getElement(ObjectID(Address));
  };

  // TODO: inline Impl
  Impl = std::make_unique<IFI>(Module.getFunction("root"),
                               GCBI,
                               *Model.get().get(),
                               CFGGetter);
  Impl->prologue();
}

void Isolate::runOnFunction(const model::Function &TheFunction) {
  llvm::Function *Function = Impl->runOnFunction(TheFunction.Entry());
  IsolatedFunctions.push_back({ TheFunction.Entry(), Function });
}

Isolate::~Isolate() {
  Impl->epilogue();

  llvm::Module &Module = Root.getModule();
  std::set<const llvm::Function *> ExternalFunctions;
  for (llvm::Function &ModuleFunction : Module.functions()) {
    if (not FunctionTags::Root.isTagOf(&ModuleFunction)
        and not FunctionTags::Isolated.isTagOf(&ModuleFunction)) {
      ExternalFunctions.insert(&ModuleFunction);
    }
  }

  llvm::Task T(IsolatedFunctions.size(),
               "Splitting functions into individual modules");
  for (auto &[Address, Function] : IsolatedFunctions) {
    T.advance(Address.toString(), true);
    std::set<const llvm::Function *> ToClone;
    ToClone.insert(Function);
    ToClone.insert(ExternalFunctions.begin(), ExternalFunctions.end());

    Output.assign(ObjectID(Address), ::cloneFiltered(Module, ToClone));
  }
}

} // namespace revng::pypeline::piperuns

void IsolateFunctions::getAnalysisUsage(llvm::AnalysisUsage &AU) const {
  AU.setPreservesAll();
  AU.addRequired<GeneratedCodeBasicInfoWrapperPass>();
  AU.addRequired<LoadModelWrapperPass>();
  AU.addRequired<ControlFlowGraphCachePass>();
  AU.addRequired<pipeline::LoadExecutionContextPass>();
}
