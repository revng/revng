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
#include "llvm/Transforms/IPO/GlobalDCE.h"
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
#include "revng/Model/Architecture.h"
#include "revng/Model/Binary.h"
#include "revng/Model/FunctionTags.h"
#include "revng/Model/NameBuilder.h"
#include "revng/Model/Register.h"
#include "revng/Support/Debug.h"
#include "revng/Support/EmitAbort.h"
#include "revng/Support/IRBuilder.h"
#include "revng/Support/IRHelpers.h"
#include "revng/Support/MetaAddress.h"
#include "revng/Support/SimplePassManager.h"

// This name is not present after `enforce-abi`.
RegisterIRHelper FDispatcher("function_dispatcher");

using namespace llvm;

static Logger TheLogger("isolation");

// Define an alias for the data structure that will contain the LLVM functions
using FunctionsMap = std::map<MDString *, Function *>;
using ValueToValueMap = DenseMap<const Value *, Value *>;

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

static void emitUnreachable(revng::IRBuilder &Builder,
                            const Twine &Reason,
                            const DebugLoc &DbgLocation) {
  // Emitting any long-lasting messages here prevents switch detection,
  // so use a simple `unreachable`.
  Builder.CreateUnreachable();
}

static void emitUnreachable(BasicBlock *BB,
                            const Twine &Reason,
                            const DebugLoc &DbgLocation) {
  revng::IRBuilder Builder(BB);
  emitUnreachable(Builder, Reason, DbgLocation);
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
  revng::pypeline::piperuns::Isolate &IP;
  GeneratedCodeBasicInfo &GCBI;
  const efa::ControlFlowGraph &FM;

public:
  CallIsolatedFunction(revng::pypeline::piperuns::Isolate &IP,
                       GeneratedCodeBasicInfo &GCBI,
                       const efa::ControlFlowGraph &FM) :
    IP(IP), GCBI(GCBI), FM(FM) {}

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
    emitUnreachable(Builder,
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
    const auto *Caller = FM.findBlock(GCBI, Builder.GetInsertBlock());

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
      CalledFunction = IP.getLocalFunction(Callee);
    else if (not SymbolName.empty())
      CalledFunction = IP.getDynamicFunction(SymbolName);
    else
      CalledFunction = IP.dispatcher();

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

namespace revng::pypeline::piperuns {

void Isolate::handleUnexpectedPCCloned(efa::OutlinedFunction &Outlined) {
  if (BasicBlock *UnexpectedPC = Outlined.UnexpectedPCCloned) {
    for (auto It = UnexpectedPC->begin(); It != UnexpectedPC->end();
         It = UnexpectedPC->begin())
      It->eraseFromParent();
    revng_assert(UnexpectedPC->empty());
    const DebugLoc &Dbg = GCBI->unexpectedPC()->getTerminator()->getDebugLoc();
    emitUnreachable(UnexpectedPC, "unexpectedPC", Dbg);
  }
}

void Isolate::handleAnyPCJumps(efa::OutlinedFunction &Outlined,
                               const efa::ControlFlowGraph &FM) {
  if (BasicBlock *AnyPC = Outlined.AnyPCCloned) {
    for (BasicBlock *AnyPCPredecessor : toVector(predecessors(AnyPC))) {
      // First of all, identify the basic block
      const efa::BasicBlock *JumpBlock = FM.findBlock(*GCBI, AnyPCPredecessor);

      Instruction *T = AnyPCPredecessor->getTerminator();
      revng_assert(not cast<BranchInst>(T)->isConditional());
      T->eraseFromParent();

      // TODO: the checks should be enabled conditionally based on the user.
      revng::IRBuilder Builder(AnyPCPredecessor);

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

} // namespace revng::pypeline::piperuns

/// Helper class that wraps `llvm::cloneModule` and specializes in the case
/// where a single function needs to be cloned. It will analyze the function to
/// be cloned and only copy across declarations of the needed functions/globals
/// needed for the body of the function to be valid. When cloning it also tries
/// to preserve some revng-specific invariants, e.g. the presence of the `PC`
/// global.
class MinimalModuleCloner {
private:
  struct FunctionUses {
    std::set<const llvm::Function *> CalledFunctions;
    std::set<const llvm::GlobalVariable *> UsedGVs;
  };

private:
  llvm::Module &Module;

  std::set<const llvm::GlobalVariable *> IgnorableGVs;

public:
  MinimalModuleCloner(const model::Binary &Binary, llvm::Module &Module) :
    Module(Module) {
    revng::verify(&Module);

    // Always copy SP because it's needed by another pipe down the pipeline
    // (InjectStackSizeProbesAtCallSitesPass). The proper fix would be to have
    // that pipe detect the absence of `SP` and assert that the stack pointer
    // could not have moved.
    // Also exclude any GV where `ProgramCounterHandler::affectsPC` returns
    // true.
    auto Architecture = Binary.Architecture();
    auto PCH = ProgramCounterHandler::fromModule(Architecture, &Module);

    // Also preserve every register CSV, later passes might want to use them
    std::set<std::string> RegisterNames;
    for (auto Register : model::Architecture::registers(Architecture))
      RegisterNames.insert(model::Register::getCSVName(Register));

    auto IsRegister = [&PCH, &RegisterNames](llvm::GlobalVariable &GV) {
      return PCH->affectsPC(&GV) or RegisterNames.contains(GV.getName().str());
    };

    // Construct the set of ignorable globals from string globals and CSVs
    // (except for special CSVs)
    for (llvm::GlobalVariable &GV : Module.globals()) {
      StringRef Name = GV.getName();
      if (Name.starts_with(DefaultStringNamespace)
          or (FunctionTags::CSV.isTagOf(&GV) and not IsRegister(GV)))
        IgnorableGVs.insert(&GV);
    }
  }

  std::unique_ptr<llvm::Module> cloneModule(llvm::Function &Function) {
    // Preserve Function and, transitively, all the KeepPostIsolation functions
    // reachable from it: those must survive isolation with their body. We
    // analyze each of them, accumulating the called functions and the used
    // global variables so the cloned module stays self-contained.
    std::set<const llvm::Function *> CalledFunctions;
    std::set<const llvm::GlobalVariable *> UsedGVs;

    OnceQueue<const llvm::Function *> Worklist;
    Worklist.insert(&Function);

    while (not Worklist.empty()) {
      const llvm::Function *Current = Worklist.pop();

      FunctionUses Uses = analyzeFunction(*Current);
      UsedGVs.insert(Uses.UsedGVs.begin(), Uses.UsedGVs.end());

      for (const llvm::Function *Callee : Uses.CalledFunctions) {
        CalledFunctions.insert(Callee);

        // Follow calls into KeepPostIsolation functions so that they, and the
        // KeepPostIsolation functions they in turn call, are preserved too.
        if (FunctionTags::KeepPostIsolation.isTagOf(Callee)
            and not Callee->isDeclaration())
          Worklist.insert(const_cast<llvm::Function *>(Callee));
      }
    }

    auto Visited = Worklist.visited();

    auto ActionFunction =
      [this, &Visited, &CalledFunctions, &UsedGVs](const GlobalValue *V) {
        auto *GV = dyn_cast<llvm::GlobalVariable>(V);
        if (GV != nullptr) {
          if (UsedGVs.contains(GV))
            return CloneAction::Clone;
          else if (IgnorableGVs.contains(GV) or GV->isDeclaration())
            return CloneAction::Omit;
          else
            return CloneAction::Clone;
        }

        auto *F = dyn_cast<llvm::Function>(V);
        if (F != nullptr) {
          if (Visited.contains(F))
            return CloneAction::Clone;
          else if (CalledFunctions.contains(F))
            return CloneAction::MakeDeclaration;
          else
            return CloneAction::Omit;
        }

        return CloneAction::Clone;
      };

    auto New = cloneFiltered(Module, ActionFunction);
    revng::verify(New.get());
    return New;
  }

private:
  FunctionUses analyzeFunction(const llvm::Function &F) {
    revng_assert(not F.isDeclaration());

    std::set<const llvm::Function *> CalledFunctions;
    std::set<const llvm::GlobalVariable *> UsedGVs;

    for (const BasicBlock &BB : F) {
      for (const llvm::Instruction &Instruction : BB) {
        // Check for call, if so recursively visit the called function
        const CallBase *CB = dyn_cast<llvm::CallBase>(&Instruction);
        if (CB != nullptr) {
          llvm::Function *CalledFunction = CB->getCalledFunction();
          if (CalledFunction != nullptr)
            CalledFunctions.insert(CalledFunction);
        }

        // Preserve all the CSVs used by helpers as well, this saves us from
        // having "surprises" later, which can be a problem since sometimes we
        // ignore non-existing CSVs.
        if (auto *Call = getCallToHelper(&Instruction)) {
          auto MaybeUsedCSVs = tryGetCSVUsedByHelperCall(Call);
          if (MaybeUsedCSVs) {
            for (auto *CSV : MaybeUsedCSVs->Read)
              UsedGVs.insert(CSV);
            for (auto *CSV : MaybeUsedCSVs->Written)
              UsedGVs.insert(CSV);
          }
        }

        std::queue<const llvm::User *> Users;
        Users.push(&Instruction);

        // Iterate the operands to find uses of global variables
        while (Users.size() > 0) {
          const llvm::User *User = Users.front();
          for (unsigned int I = 0; I < User->getNumOperands(); I++) {
            Value *Operand = User->getOperand(I);

            GlobalVariable *GV = dyn_cast<GlobalVariable>(Operand);
            if (GV != nullptr) {
              UsedGVs.insert(GV);
              continue;
            }

            ConstantExpr *CE = dyn_cast<ConstantExpr>(Operand);
            if (CE != nullptr) {
              Users.push(CE);
              continue;
            }
          }
          Users.pop();
        }
      }
    }

    return { std::move(CalledFunctions), std::move(UsedGVs) };
  }
};

namespace revng::pypeline::piperuns {

Isolate::Isolate(const class Model &Model,
                 llvm::StringRef Config,
                 llvm::StringRef DynamicConfig,
                 const CFGMap &CFG,
                 LLVMRootContainer &Root,
                 LLVMFunctionContainer &Output) :
  Binary(*Model.get().get()), CFG(CFG), Output(Output) {
  // Manually perform `cloneIntoContext` to prune the root container as early as
  // possible
  llvm::SmallVector<char, 0> Buffer;
  writeBitcode(Root.getModule(), Buffer);
  Root.disposeIfPossible();
  ClonedModule = readBitcode(Buffer, Output.getContext());

  llvm::LLVMContext &Context = ClonedModule->getContext();
  IsolatedFunctionType = createFunctionType<void>(Context);
  GCBI.emplace(*Model.get().get(), *ClonedModule);

  auto SimpleFunctionType = createFunctionType<void>(Context);
  FunctionDispatcher = createIRHelper("function_dispatcher",
                                      *ClonedModule,
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
                                         *ClonedModule);
    FunctionTags::DynamicFunction.addTo(NewFunction);
    NewFunction->addFnAttr(Attribute::NoMerge);
    DynamicFunctionsMap[Name] = NewFunction;
  }
}

void Isolate::runOnFunction(const model::Function &TheFunction) {
  const MetaAddress Entry = TheFunction.Entry();
  revng_assert(Entry.isValid());
  const efa::ControlFlowGraph &FM = *CFG.getElement(ObjectID(Entry));

  // Get or create the llvm::Function
  Function *F = getLocalFunction(Entry);

  // Decorate the function as appropriate
  F->addFnAttr(Attribute::NullPointerIsValid);
  F->addFnAttr(Attribute::NoMerge);
  IsolatedFunctionsMap[Entry] = F;
  revng_assert(F != nullptr);

  // Outline the function (later on we'll steal its body and move it into F)
  CallIsolatedFunction CallHandler(*this, *GCBI, FM);
  FunctionOutliner Outliner(*ClonedModule, Binary, *GCBI);
  efa::OutlinedFunction Outlined = Outliner.outline(Entry, &CallHandler);

  handleUnexpectedPCCloned(Outlined);

  handleAnyPCJumps(Outlined, FM);

  if (Outlined.Function)
    for (BasicBlock &BB : *Outlined.Function)
      revng_assert(BB.getTerminator() != nullptr);

  // Steal the function body and let the outlined function be destroyed
  moveBlocksInto(*Outlined.Function, *F);

  // Store the isolated function for later splitting
  IsolatedFunctions.push_back({ TheFunction.Entry(), F });
}

Function *Isolate::getLocalFunction(const MetaAddress &Entry) {
  auto Name = llvmName(Binary.Functions().at(Entry));
  if (auto *F = ClonedModule->getFunction(Name))
    return F;

  auto *F = Function::Create(IsolatedFunctionType,
                             GlobalValue::ExternalLinkage,
                             Name,
                             *ClonedModule);
  FunctionTags::Isolated.addTo(F);
  setMetaAddressMetadata(F, FunctionEntryMDName, Entry);
  return F;
}

Function *Isolate::getDynamicFunction(llvm::StringRef SymbolName) const {
  return DynamicFunctionsMap.at(SymbolName);
}

void Isolate::splitIsolatedFunctionsToOutput() {
  // Run globaldce on the isolated root module to remove any cruft present. This
  // should speed up cloning.
  SimplePassManager PM;
  PM.addPass(llvm::GlobalDCEPass());
  PM.run(*ClonedModule);

  llvm::Task T(IsolatedFunctions.size(),
               "Splitting functions into individual modules");

  MinimalModuleCloner Cloner(Binary, *ClonedModule);

  for (auto &[Address, Function] : IsolatedFunctions) {
    T.advance(Address.toString(), true);

    auto IsolatedModule = Cloner.cloneModule(*Function);
    // Check that running globaldce on the isolated module does nothing
    if (VerifyLog.isEnabled()) {
      llvm::StringSet GlobalNames;
      for (GlobalValue &GV : IsolatedModule->global_values()) {
        if (GV.hasName())
          GlobalNames.insert(GV.getName());
      }

      SimplePassManager PM;
      PM.addPass(llvm::GlobalDCEPass());
      PM.run(*IsolatedModule);

      for (GlobalValue &GV : IsolatedModule->global_values()) {
        if (GV.hasName())
          GlobalNames.erase(GV.getName());
      }
      revng_assert(GlobalNames.size() == 0,
                   "GlobalDCE removed names from an isolated module");
    }
    Output.assign(ObjectID(Address), std::move(IsolatedModule));

    // Since we saved the function to the output, delete its body in
    // ClonedModule to save memory
    deleteOnlyBody(*Function);
  }
}

Isolate::~Isolate() {
  llvm::Task T(4, "Isolate: epilogue");
  T.advance("Verify module", true);
  revng::verify(ClonedModule.get());

  // Drop the `root` function's body to save memory, since we're done isolating
  T.advance("Cleanup", true);
  deleteOnlyBody(*ClonedModule->getFunction("root"));

  // Before emitting it in output, verify the module
  T.advance("Verify module", true);
  revng::verify(ClonedModule.get());

  // Actually split the isolated function to the output module
  T.advance("Split isolated functions", true);
  splitIsolatedFunctionsToOutput();
}

} // namespace revng::pypeline::piperuns
