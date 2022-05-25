/// \file IsolateFunctions.cpp
/// \brief Implements the IsolateFunctions pass which applies function isolation
///        using the informations provided by EarlyFunctionAnalysis.

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/IR/DIBuilder.h"
#include "llvm/IR/DebugInfoMetadata.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Support/raw_os_ostream.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/Transforms/Utils/CodeExtractor.h"
#include "llvm/Transforms/Utils/Local.h"

#include "revng/ADT/KeyedObjectContainer.h"
#include "revng/ADT/KeyedObjectTraits.h"
#include "revng/ADT/ZipMapIterator.h"
#include "revng/BasicAnalyses/GeneratedCodeBasicInfo.h"
#include "revng/EarlyFunctionAnalysis/IRHelpers.h"
#include "revng/FunctionIsolation/IsolateFunctions.h"
#include "revng/Model/Binary.h"
#include "revng/Pipeline/AllRegistries.h"
#include "revng/Pipeline/Contract.h"
#include "revng/Pipes/Kinds.h"
#include "revng/Pipes/RootKind.h"
#include "revng/Pipes/TaggedFunctionKind.h"
#include "revng/Support/Debug.h"
#include "revng/Support/FunctionTags.h"
#include "revng/Support/IRHelpers.h"

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
    using namespace pipeline;
    using namespace ::revng::pipes;
    return {
      ContractGroup::transformOnlyArgument(Root,
                                           Exactness::Exact,
                                           Isolated,
                                           InputPreservation::Preserve)
    };
  }

  void registerPasses(llvm::legacy::PassManager &Manager) {
    Manager.add(new IsolateFunctions());
  }
};

static pipeline::RegisterLLVMPass<IsolatePipe> Y;

static void
eraseBranch(Instruction *I, BasicBlock *ExpectedUniqueSuccessor = nullptr) {
  auto *T = cast<BranchInst>(I);
  revng_assert(T->isUnconditional());
  if (ExpectedUniqueSuccessor != nullptr)
    revng_assert(T->getSuccessor(0) == ExpectedUniqueSuccessor);
  eraseFromParent(T);
}

class ConstantStringsPool {
private:
  Module *M;
  std::map<std::string, GlobalVariable *> StringsPool;

public:
  ConstantStringsPool(Module *M) : M(M) {}

  Constant *get(std::string String, const Twine &Name = "") {
    auto It = StringsPool.find(String);
    auto &C = M->getContext();
    if (It == StringsPool.end()) {
      auto *Initializer = ConstantDataArray::getString(C, String, true);
      auto *NewVariable = new GlobalVariable(*M,
                                             Initializer->getType(),
                                             true,
                                             GlobalValue::InternalLinkage,
                                             Initializer);
      It = StringsPool.insert(It, { String, NewVariable });
    }

    auto *U8PtrTy = Type::getInt8Ty(C)->getPointerTo();
    return ConstantExpr::getPointerCast(It->second, U8PtrTy);
  }
};

using SuccessorsList = GeneratedCodeBasicInfo::SuccessorsList;
struct Boundary {
  BasicBlock *Block = nullptr;
  BasicBlock *CalleeBlock = nullptr;
  BasicBlock *ReturnBlock = nullptr;
  SuccessorsList Successors;

  bool isCall() const { return ReturnBlock != nullptr; }

  void dump() const debug_function { dump(dbg); }

  template<typename O>
  void dump(O &Output) const {
    Output << "Block: " << getName(Block) << "\n";
    Output << "CalleeBlock: " << getName(CalleeBlock) << "\n";
    Output << "ReturnBlock: " << getName(ReturnBlock) << "\n";
    Output << "Successors: \n";
    Successors.dump(Output);
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
  Function *RaiseException = nullptr;
  Function *FunctionDispatcher = nullptr;
  std::map<MetaAddress, Function *> IsolatedFunctionsMap;
  std::map<StringRef, Function *> DynamicFunctionsMap;
  ConstantStringsPool Strings;
  GlobalVariable *ExceptionSourcePC;
  GlobalVariable *ExceptionDestinationPC;

public:
  IsolateFunctionsImpl(Function *RootFunction,
                       GeneratedCodeBasicInfo &GCBI,
                       const model::Binary &Binary) :
    RootFunction(RootFunction),
    TheModule(RootFunction->getParent()),
    Context(TheModule->getContext()),
    GCBI(GCBI),
    Binary(Binary),
    Strings(TheModule) {}

  void run();

private:
  /// Isolate the function described by \p Function
  void isolate(const model::Function &Function);

  /// Process a basic block from the model
  void handleBasicBlock(const efa::BasicBlock &Block,
                        ValueToValueMapTy &OldToNew,
                        FunctionBlocks &ClonedBlocks);

  /// Clone the basic blocks involved in \p Entry jump target
  ///
  /// \return a vector of boundary basic blocks
  std::vector<Boundary>
  cloneAndIdentifyBoundaries(const efa::BasicBlock &Block,
                             ValueToValueMapTy &OldToNew,
                             FunctionBlocks &ClonedBlocks);

  /// Create the code necessary to handle a direct branch in the IR
  bool handleDirectBoundary(const Boundary &TheBoundary,
                            const efa::BasicBlock &Block,
                            SuccessorsContainer &ExpectedSuccessors,
                            FunctionBlocks &ClonedBlocks);

  /// Create the code necessary to handle an indirect branch in the IR
  bool handleIndirectBoundary(const std::vector<Boundary> &Boundaries,
                              const efa::BasicBlock &Block,
                              const SuccessorsContainer &ExpectedSuccessors,
                              bool CallConsumed,
                              FunctionBlocks &ClonedBlocks);

  /// Emit a function call marker and a branch to the return address
  void createFunctionCall(IRBuilder<> &Builder,
                          Function *Callee,
                          const Boundary &TheBoundary,
                          const MetaAddress &CallerBlockAddress,
                          bool IsNoReturn);

  void createFunctionCall(IRBuilder<> &Builder,
                          MetaAddress ExpectedCallee,
                          const Boundary &TheBoundary,
                          const MetaAddress &CallerBlockAddress,
                          bool IsNoReturn);

  void createFunctionCall(BasicBlock *BB,
                          Function *Callee,
                          const Boundary &TheBoundary,
                          const MetaAddress &CallerBlockAddress,
                          bool IsNoReturn) {
    IRBuilder<> Builder(BB);
    createFunctionCall(Builder,
                       Callee,
                       TheBoundary,
                       CallerBlockAddress,
                       IsNoReturn);
  }

  void createFunctionCall(BasicBlock *BB,
                          MetaAddress Callee,
                          const Boundary &TheBoundary,
                          const MetaAddress &CallerBlockAddress,
                          bool IsNoReturn) {
    IRBuilder<> Builder(BB);
    createFunctionCall(Builder,
                       Callee,
                       TheBoundary,
                       CallerBlockAddress,
                       IsNoReturn);
  }

  /// Post process all the call markers, replacing them with actual calls
  void replaceCallMarker() const;

  /// Populate the function_dispatcher, needed to handle the indirect calls
  void populateFunctionDispatcher();

  /// Create code to throw of an exception
  void throwException(IRBuilder<> &Builder,
                      const Twine &Reason,
                      const DebugLoc &DbgLocation);

  void throwException(BasicBlock *BB,
                      const Twine &Reason,
                      const DebugLoc &DbgLocation) {
    IRBuilder<> Builder(BB);
    throwException(Builder, Reason, DbgLocation);
  }
};

void IFI::throwException(IRBuilder<> &Builder,
                         const Twine &Reason,
                         const DebugLoc &DbgLocation) {
  revng_assert(RaiseException != nullptr);
  // revng_assert(DbgLocation);

  // Create the message string
  Constant *ReasonString = Strings.get(Reason.str());

  // Populate the source PC
  MetaAddress SourcePC = MetaAddress::invalid();

  if (Instruction *T = Builder.GetInsertBlock()->getTerminator())
    SourcePC = getPC(T).first;

  auto *Ty = ExceptionSourcePC->getType()->getPointerElementType();
  Builder.CreateStore(SourcePC.toConstant(Ty), ExceptionSourcePC);

  // Populate the destination PC
  Builder.CreateStore(GCBI.programCounterHandler()->loadPC(Builder),
                      ExceptionDestinationPC);

  auto *NewCall = Builder.CreateCall(RaiseException,
                                     { ReasonString,
                                       ExceptionSourcePC,
                                       ExceptionDestinationPC });
  NewCall->setDebugLoc(DbgLocation);
  Builder.CreateUnreachable();
}

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
  throwException(Unexpected, "An unexpected functions has been called", Dbg);
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
static auto
zeroOrOne(const T &Range, const F &Predicate) -> decltype(&*Range.begin()) {
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

static bool isDirectEdge(efa::FunctionEdgeType::Values Type) {
  using namespace efa::FunctionEdgeType;

  switch (Type) {
  case IndirectCall:
  case Return:
  case BrokenReturn:
  case IndirectTailCall:
  case LongJmp:
  case Killer:
  case Unreachable:
    return false;

  case DirectBranch:
  case FakeFunctionCall:
  case FunctionCall:
  case FakeFunctionReturn:
    return true;

  case Invalid:
  case Count:
    revng_abort();
  }
}

static bool isIndirectEdge(efa::FunctionEdgeType::Values Type) {
  return not isDirectEdge(Type);
}

template<typename LeftMap, typename RightMap>
void printAddressListComparison(const LeftMap &ExpectedAddresses,
                                const RightMap &ActualAddresses) {
  // Compare expected and actual
  if (TheLogger.isEnabled()) {
    for (auto [ExpectedAddress, ActualAddress] :
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

bool IFI::handleIndirectBoundary(const std::vector<Boundary> &Boundaries,
                                 const efa::BasicBlock &Block,
                                 const SuccessorsContainer &ExpectedSuccessors,
                                 bool CallConsumed,
                                 FunctionBlocks &ClonedBlocks) {
  std::vector<const efa::FunctionEdgeBase *> RemainingEdges;
  for (auto &[Edge, EdgeUsageCount] : ExpectedSuccessors)
    if (EdgeUsageCount == 0)
      RemainingEdges.push_back(Edge);

  // At this point RemainingEdges must either be a series of
  // `DirectBranch` or a single one of the indirect ones
  bool NoMore = RemainingEdges.size() == 0;
  using namespace efa::FunctionEdgeType;

  auto IsDirectEdge = [](const efa::FunctionEdgeBase *E) {
    return isDirectEdge(E->Type);
  };
  bool AllDirect = allOrNone(RemainingEdges, IsDirectEdge, false);

  auto IndirectType = Invalid;
  if (not NoMore and not AllDirect) {
    // We're not out of expected successors, but they are not all direct.
    // We expect a single indirect edge.
    revng_assert(RemainingEdges.size() == 1);
    IndirectType = (*RemainingEdges.begin())->Type;
    revng_assert(isIndirectEdge(IndirectType));
  }

  // We now consumed all the direct jump/calls, let's proceed with the
  // indirect jump/call
  auto IsIndirectBoundary = [](const Boundary &B) {
    const auto &S = B.Successors;
    return S.AnyPC or S.UnexpectedPC or not S.hasSuccessors();
  };
  const Boundary *IndirectBoundary = zeroOrOne(Boundaries, IsIndirectBoundary);

  if (IndirectBoundary == nullptr) {
    // Is there a leftover direct boundary? If so, assign it anyways
    if (not NoMore and Boundaries.size()) {
      for (const Boundary &B : Boundaries)
        if (isFunctionCall(B.Block))
          IndirectBoundary = &B;
    } else {
      revng_assert(NoMore);
      return false;
    }
  }

  BasicBlock *BB = IndirectBoundary->Block;

  // Whatever the situation, we need to replace the terminator of this basic
  // block at this point
  Instruction *OldTerminator = BB->getTerminator();
  IRBuilder<> Builder(OldTerminator);

  if (AllDirect) {
    if (IndirectBoundary->isCall()) {
      if (TheLogger.isEnabled()) {
        TheLogger << "The model expects a set of direct branches from ";
        Block.End.dump(TheLogger);
        TheLogger << ", but in the binary a function call has been identified."
                  << DoLog;
      }
    }

    SortedVector<MetaAddress> ExpectedAddresses;
    {
      auto Inserter = ExpectedAddresses.batch_insert();
      for (const efa::FunctionEdgeBase *Edge : RemainingEdges)
        Inserter.insert(Edge->Destination);
    }

    // Print comparison of targets mandated by the model and those identified in
    // the IR
    printAddressListComparison(ExpectedAddresses,
                               IndirectBoundary->Successors.Addresses);

    // Create the dispatcher for the targets
    auto Dispatcher = GCBI.buildDispatcher(ExpectedAddresses,
                                           Builder,
                                           ClonedBlocks.unexpectedPCBlock());
    for (BasicBlock *BB : Dispatcher.NewBlocks)
      ClonedBlocks.push_back(BB);

  } else if (not NoMore) {
    if (TheLogger.isEnabled()
        and IndirectBoundary->isCall() != (IndirectType == IndirectCall)) {
      TheLogger << "The model, at ";
      Block.End.dump(TheLogger);
      TheLogger << ", ";
      if (IndirectType == IndirectCall)
        TheLogger << "expects an indirect call, but it's not";
      else
        TheLogger << "does not expect an indirect call, but it's";
      TheLogger << " in the binary." << DoLog;
    }

    switch (IndirectType) {
    case IndirectCall:
    case IndirectTailCall:
      createFunctionCall(Builder,
                         MetaAddress::invalid(),
                         *IndirectBoundary,
                         Block.Start,
                         false);
      break;

    case Return:
      Builder.CreateBr(ClonedBlocks.returnBlock());
      break;

    case BrokenReturn: {
      BasicBlock::iterator InsertPoint = Builder.GetInsertPoint();
      revng_assert(not Builder.GetInsertBlock()->empty());
      Instruction *Old = InsertPoint == Builder.GetInsertBlock()->end() ?
                           &*Builder.GetInsertBlock()->rbegin() :
                           &*InsertPoint;
      throwException(Builder, "A broken return was taken", Old->getDebugLoc());
    } break;
    case LongJmp: {
      BasicBlock::iterator InsertPoint = Builder.GetInsertPoint();
      revng_assert(not Builder.GetInsertBlock()->empty());
      Instruction *Old = InsertPoint == Builder.GetInsertBlock()->end() ?
                           &*Builder.GetInsertBlock()->rbegin() :
                           &*InsertPoint;
      throwException(Builder, "A longjmp was taken", Old->getDebugLoc());
    } break;
    case Killer: {
      BasicBlock::iterator InsertPoint = Builder.GetInsertPoint();
      revng_assert(not Builder.GetInsertBlock()->empty());
      Instruction *Old = InsertPoint == Builder.GetInsertBlock()->end() ?
                           &*Builder.GetInsertBlock()->rbegin() :
                           &*InsertPoint;
      throwException(Builder,
                     "A killer block has been reached",
                     Old->getDebugLoc());
    } break;
    case Unreachable: {
      BasicBlock::iterator InsertPoint = Builder.GetInsertPoint();
      revng_assert(not Builder.GetInsertBlock()->empty());
      Instruction *Old = InsertPoint == Builder.GetInsertBlock()->end() ?
                           &*Builder.GetInsertBlock()->rbegin() :
                           &*InsertPoint;
      throwException(Builder,
                     "An unrechable instruction has been "
                     "reached",
                     Old->getDebugLoc());
    } break;

    default:
      revng_abort();
    }
  } else {
    // We have an indirect boundary but the model does not expect it

    if (TheLogger.isEnabled()) {
      TheLogger << "The model, at ";
      Block.End.dump(TheLogger);
      TheLogger << ", does not expect an indirect branch, ";
      TheLogger << "but it's in the binary." << DoLog;
    }

    Builder.CreateBr(ClonedBlocks.unexpectedPCBlock());
  }

  eraseFromParent(OldTerminator);

  return true;
}

/// \return true if this was a call
bool IFI::handleDirectBoundary(const Boundary &TheBoundary,
                               const efa::BasicBlock &Block,
                               SuccessorsContainer &ExpectedSuccessors,
                               FunctionBlocks &ClonedBlocks) {
  BasicBlock *BB = TheBoundary.Block;
  SetAtMostOnce IsCall;
  size_t Consumed = 0;
  for (auto &[Edge, EdgeUsageCount] : ExpectedSuccessors) {
    // Is this edge targeting who we expect?
    if (TheBoundary.Successors.Addresses.count(Edge->Destination) == 0)
      continue;

    bool Match = false;
    switch (Edge->Type) {
    case efa::FunctionEdgeType::DirectBranch:
      if (not TheBoundary.isCall())
        Match = true;
      break;

    case efa::FunctionEdgeType::FunctionCall:
    case efa::FunctionEdgeType::FakeFunctionCall:
      if (TheBoundary.isCall())
        Match = true;
      break;

    default:
      break;
    }

    if (not Match)
      continue;

    ++Consumed;
    ++EdgeUsageCount;

    if (TheBoundary.isCall()) {
      IsCall.set();

      if (Edge->Type == efa::FunctionEdgeType::FunctionCall) {
        auto *Call = cast<efa::CallEdge>(Edge);
        eraseBranch(BB->getTerminator(), TheBoundary.CalleeBlock);
        createFunctionCall(BB,
                           Edge->Destination,
                           TheBoundary,
                           Block.Start,
                           hasAttribute(Binary,
                                        *Call,
                                        model::FunctionAttribute::NoReturn));
      }
    }
  }

  // Under certain conditions, we might have a direct boundary that does not
  // correspond to any direct `efa::FunctionEdgeBase`. An example is when we
  // have a direct call to a function that is not available in the model. In
  // fact, in this case, EFA will emit an `efa::FunctionEdgeBase` representing
  // and *Indirect*Call.
  revng_assert(Consumed <= TheBoundary.Successors.Addresses.size());

  return IsCall;
}

std::vector<Boundary>
IFI::cloneAndIdentifyBoundaries(const efa::BasicBlock &Block,
                                ValueToValueMapTy &OldToNew,
                                FunctionBlocks &ClonedBlocks) {
  MetaAddress Entry = Block.Start;
  std::set<BasicBlock *> Blocks;
  for (BasicBlock *Block : GCBI.getBlocksGeneratedByPC(Entry))
    Blocks.insert(Block);
  revng_assert(Blocks.size() > 0);

  std::vector<Boundary> Boundaries;

  for (BasicBlock *BB : Blocks) {
    // Clone basic block in root and register it
    auto *NewBB = CloneBasicBlock(BB, OldToNew, "", RootFunction);
    revng_assert(OldToNew.count(BB) == 0);
    OldToNew.insert({ BB, NewBB });
    ClonedBlocks.push_back(NewBB);

    // Is this a boundary basic block?
    auto HasNotBeenCloned = [&Blocks](BasicBlock *Successor) {
      return Blocks.count(Successor) == 0;
    };
    if (succ_empty(BB) or llvm::any_of(successors(BB), HasNotBeenCloned)) {
      BasicBlock *Callee = getFunctionCallCallee(BB);
      BasicBlock *Fallthrough = getFallthrough(BB);
      Boundary NewBoundary{
        NewBB, Callee, Fallthrough, GCBI.getSuccessors(BB)
      };
      Boundaries.push_back(NewBoundary);
    }
  }

  if (TheLogger.isEnabled()) {
    TheLogger << "Boundaries: \n";
    for (const Boundary &B : Boundaries) {
      B.dump(TheLogger);
      TheLogger << "\n";
    }
    TheLogger << DoLog;
    TheLogger << "Expected:\n";
    std::string Buffer;
    {
      raw_string_ostream StringStream(Buffer);
      yaml::Output YAMLOutput(StringStream);
      for (auto Edge : Block.Successors) {
        YAMLOutput << Edge;
      }
    }
    TheLogger << Buffer << DoLog;
  }

  return Boundaries;
}

void IFI::handleBasicBlock(const efa::BasicBlock &Block,
                           ValueToValueMapTy &OldToNew,
                           FunctionBlocks &ClonedBlocks) {
  if (TheLogger.isEnabled()) {
    TheLogger << "Isolating ";
    Block.Start.dump(TheLogger);
    TheLogger << "-";
    Block.End.dump(TheLogger);
    TheLogger << DoLog;
  }

  LoggerIndent<> Indent(TheLogger);

  // Sentinel to ensure we don't have more than a call within a basic block
  SetAtMostOnce CallConsumed;

  // Identify boundary blocks
  std::vector<Boundary> Boundaries = cloneAndIdentifyBoundaries(Block.Start,
                                                                OldToNew,
                                                                ClonedBlocks);

  // Handle call to dynamic functions
  efa::CallEdge *Call = nullptr;
  for (const auto &Edge : Block.Successors)
    if ((Call = dyn_cast<efa::CallEdge>(Edge.get())))
      break;

  if (Call != nullptr and not Call->DynamicFunction.empty()) {
    revng_assert(Boundaries.size() == 1);
    const auto &TheBoundary = Boundaries[0];
    auto *BB = TheBoundary.Block;

    eraseBranch(BB->getTerminator(), TheBoundary.CalleeBlock);

    createFunctionCall(BB,
                       DynamicFunctionsMap.at(Call->DynamicFunction),
                       TheBoundary,
                       Block.Start,
                       hasAttribute(Binary,
                                    *Call,
                                    model::FunctionAttribute::NoReturn));

    return;
  }

  // At this point, we first need to handle all the boundary blocks that
  // represent direct jumps, then we'll take care of the (only) indirect jump,
  // if any

  SuccessorsContainer ExpectedSuccessors;
  for (const auto &E : Block.Successors) {
    // Ignore self-loops
    if (E->Destination != Block.Start) {
      ExpectedSuccessors[E.get()] = 0;
    }
  }

  // Consume direct jumps calls
  for (const auto &Boundary : Boundaries) {
    if (not(Boundary.Successors.AnyPC or Boundary.Successors.UnexpectedPC)) {
      bool Result = handleDirectBoundary(Boundary,
                                         Block,
                                         ExpectedSuccessors,
                                         ClonedBlocks);
      CallConsumed.setIf(Result);
    }
  }

  bool HasIndirectBoundary = handleIndirectBoundary(Boundaries,
                                                    Block,
                                                    ExpectedSuccessors,
                                                    CallConsumed,
                                                    ClonedBlocks);

  // TODO: this is obscure and confusing
  revng_assert(not(HasIndirectBoundary and CallConsumed));
}

void IFI::isolate(const model::Function &Function) {
  // Map from original values to new ones
  ValueToValueMapTy OldToNew;

  // List of cloned basic blocks, dummy entry and return block are preallocated
  FunctionBlocks ClonedBlocks;

  auto CreateBB = [this](StringRef Name) {
    return BasicBlock::Create(Context, Name, RootFunction, nullptr);
  };

  // Create return block
  ClonedBlocks.returnBlock() = CreateBB("return");
  ReturnInst::Create(Context, ClonedBlocks.returnBlock());

  // Create unexpectedPC block
  ClonedBlocks.unexpectedPCBlock() = CreateBB("unexpectedPC");
  const DebugLoc &Dbg = GCBI.unexpectedPC()->getTerminator()->getDebugLoc();
  throwException(ClonedBlocks.unexpectedPCBlock(), "unexpectedPC", Dbg);
  OldToNew[GCBI.unexpectedPC()] = ClonedBlocks.unexpectedPCBlock();

  // Get the entry basic block
  BasicBlock *OriginalEntry = GCBI.getBlockAt(Function.Entry);

  TheLogger << "Isolating ";
  Function.Entry.dump(TheLogger);
  TheLogger << DoLog;
  LoggerIndent<> Indent(TheLogger);

  // The CFG must exist if the type of function is not `Invalid`
  auto *Term = OriginalEntry->getTerminator();
  auto *FMMDNode = Term->getMetadata(FunctionMetadataMDName);
  revng_assert(FMMDNode && Function.Type != model::FunctionType::Invalid);

  // Extract function metadata
  efa::FunctionMetadata FM = *extractFunctionMetadata(OriginalEntry).get();

  // Process each basic block
  for (const efa::BasicBlock &Block : FM.ControlFlowGraph)
    handleBasicBlock(Block, OldToNew, ClonedBlocks);

  // Create a dummy entry branching to real entry
  revng_assert(ClonedBlocks.dummyEntryBlock() == nullptr);
  ClonedBlocks.dummyEntryBlock() = CreateBB("dummyentry");
  BranchInst::Create(cast<BasicBlock>(&*OldToNew[OriginalEntry]),
                     ClonedBlocks.dummyEntryBlock());

  // Drop all calls to `function_call`
  std::vector<Instruction *> ToDrop;
  for (BasicBlock *BB : ClonedBlocks)
    for (Instruction &I : *BB)
      if (isCallTo(&I, "function_call"))
        ToDrop.push_back(&I);

  for (Instruction *I : ToDrop)
    eraseFromParent(I);

  remapInstructionsInBlocks(ClonedBlocks.Blocks, OldToNew);

  // Let CodeExtractor create the new function
  // TODO: can we hoist CEAC?
  CodeExtractorAnalysisCache CEAC(*RootFunction);
  CodeExtractor CE(ClonedBlocks.Blocks,
                   nullptr,
                   false,
                   nullptr,
                   nullptr,
                   nullptr,
                   false,
                   true,
                   "");
  llvm::Function *NewFunction = CE.extractCodeRegion(CEAC);
  revng_assert(NewFunction != nullptr);

  FunctionType *FT = NewFunction->getFunctionType();
  revng_assert(FT->getReturnType()->isVoidTy());
  revng_assert(FT->getNumParams() == 0);

  auto *TargetFunction = IsolatedFunctionsMap.at(Function.Entry);

  // Record all the blocks
  SmallVector<BasicBlock *, 16> Blocks;
  for (BasicBlock &BB : *NewFunction)
    Blocks.push_back(&BB);

  // Move the blocks
  for (BasicBlock *BB : Blocks) {
    BB->removeFromParent();
    TargetFunction->getBasicBlockList().push_back(BB);
  }

  // Drop the temporary isolated functions and its call
  auto UserIt = NewFunction->user_begin();
  revng_assert(UserIt != NewFunction->user_end());
  auto *Call = cast<CallInst>(*UserIt);
  ++UserIt;
  revng_assert(UserIt == NewFunction->user_end());
  eraseFromParent(Call);

  revng_assert(NewFunction->use_empty());
  revng_assert(NewFunction->getBasicBlockList().empty());
  eraseFromParent(NewFunction);
}

void IFI::createFunctionCall(IRBuilder<> &Builder,
                             MetaAddress ExpectedCallee,
                             const Boundary &TheBoundary,
                             const MetaAddress &CallerBlockAddress,
                             bool IsNoReturn) {
  Function *Callee = nullptr;
  BasicBlock *ExpectedCalleeBB = nullptr;
  if (ExpectedCallee.isValid()) {
    Callee = IsolatedFunctionsMap.at(ExpectedCallee);
    ExpectedCalleeBB = GCBI.getBlockAt(ExpectedCallee);
  }

  if (TheBoundary.CalleeBlock != nullptr
      and TheBoundary.CalleeBlock != ExpectedCalleeBB) {
    revng_log(TheLogger,
              "Warning: The callee in the binary ("
                << getName(TheBoundary.CalleeBlock)
                << ") is different from the one provided by the model ("
                << getName(ExpectedCalleeBB) << ")");
  }

  createFunctionCall(Builder,
                     Callee,
                     TheBoundary,
                     CallerBlockAddress,
                     IsNoReturn);
}

void IFI::createFunctionCall(IRBuilder<> &Builder,
                             Function *Callee,
                             const Boundary &TheBoundary,
                             const MetaAddress &CallerBlockAddress,
                             bool IsNoReturn) {

  if (Callee == nullptr)
    Callee = FunctionDispatcher;

  BasicBlock::iterator InsertPoint = Builder.GetInsertPoint();
  revng_assert(not Builder.GetInsertBlock()->empty());
  bool AtEnd = InsertPoint == Builder.GetInsertBlock()->end();
  Instruction *Old = AtEnd ? &*Builder.GetInsertBlock()->rbegin() :
                             &*InsertPoint;
  auto *NewCall = Builder.CreateCall(Callee);
  NewCall->setDebugLoc(Old->getDebugLoc());
  FunctionTags::CallToLifted.addTo(NewCall);
  GCBI.setMetaAddressMetadata(NewCall,
                              CallerBlockStartMDName,
                              CallerBlockAddress);

  if (IsNoReturn) {
    throwException(Builder,
                   "We return from a noreturn function call",
                   Old->getDebugLoc());
  } else if (TheBoundary.ReturnBlock != nullptr) {
    // Emit jump to fallthrough
    Builder.CreateBr(TheBoundary.ReturnBlock);
  } else {
    if (TheLogger.isEnabled()) {
      TheLogger << "Call to " << Callee->getName() << " in "
                << getName(TheBoundary.Block)
                << " has not been detected as a function call in the binary."
                << DoLog;
    }

    throwException(Builder,
                   "An instruction marked as a call has not been "
                   "identified as such in the binary",
                   Old->getDebugLoc());
  }
}

void IFI::run() {
  ExceptionSourcePC = MetaAddress::createStructVariable(TheModule,
                                                        "exception_source_pc");
  ExceptionDestinationPC = MetaAddress::createStructVariable(TheModule,
                                                             "exception_"
                                                             "destination_pc");

  // Declare the raise_exception_helper function that we will use as a throw
  std::vector<Type *> ArgsType{ Type::getInt8Ty(Context)->getPointerTo(),
                                ExceptionSourcePC->getType(),
                                ExceptionDestinationPC->getType() };
  auto *RaiseExceptionTy = FunctionType::get(Type::getVoidTy(Context),
                                             ArgsType,
                                             false);
  RaiseException = Function::Create(RaiseExceptionTy,
                                    Function::ExternalLinkage,
                                    "raise_exception_helper",
                                    TheModule);
  FunctionTags::Exceptional.addTo(RaiseException);

  FunctionDispatcher = Function::Create(createFunctionType<void>(Context),
                                        GlobalValue::ExternalLinkage,
                                        "function_dispatcher",
                                        TheModule);
  FunctionTags::FunctionDispatcher.addTo(FunctionDispatcher);

  auto *IsolatedFunctionType = createFunctionType<void>(Context);

  // Create all the dynamic functions
  for (const model::DynamicFunction &Function :
       Binary.ImportedDynamicFunctions) {
    StringRef Name = Function.OriginalName;
    auto *NewFunction = Function::Create(IsolatedFunctionType,
                                         GlobalValue::ExternalLinkage,
                                         "dynamic_" + Function.OriginalName,
                                         TheModule);
    FunctionTags::DynamicFunction.addTo(NewFunction);

    auto *EntryBB = BasicBlock::Create(Context, "", NewFunction);
    throwException(EntryBB, Twine("Dynamic call ") + Name, DebugLoc());

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

  // Precreate all the isolated functions
  for (const model::Function &Function : Binary.Functions) {
    if (Function.Type == model::FunctionType::Fake)
      continue;

    auto *NewFunction = Function::Create(IsolatedFunctionType,
                                         GlobalValue::ExternalLinkage,
                                         "local_" + Function.name(),
                                         TheModule);
    NewFunction->addFnAttr(Attribute::NullPointerIsValid);
    IsolatedFunctionsMap[Function.Entry] = NewFunction;
    FunctionTags::Isolated.addTo(NewFunction);
    revng_assert(NewFunction != nullptr);
    GCBI.setMetaAddressMetadata(NewFunction,
                                FunctionEntryMDNName,
                                Function.Entry);

    auto *OriginalEntryTerm = GCBI.getBlockAt(Function.Entry)->getTerminator();
    auto *MDNode = OriginalEntryTerm->getMetadata(FunctionMetadataMDName);
    NewFunction->setMetadata(FunctionMetadataMDName, MDNode);
  }

  std::set<Function *> IsolatedFunctions;
  for (const model::Function &Function : Binary.Functions) {
    // Do not isolate fake functions
    if (Function.Type == model::FunctionType::Fake)
      continue;

    // Perform isolation
    isolate(Function);
  }

  revng_check(not verifyModule(*TheModule, &dbgs()));

  // Create the functions and basic blocks needed for the correct execution of
  // the exception handling mechanism

  // Populate the function_dispatcher
  populateFunctionDispatcher();

  // Cleanup root
  EliminateUnreachableBlocks(*RootFunction, nullptr, false);

  // Before emitting it in output we check that the module in passes the
  // verifyModule pass
  if (VerifyLog.isEnabled())
    revng_assert(not verifyModule(*TheModule, &dbgs()));

  FunctionTags::IsolatedRoot.addTo(RootFunction);
}

bool IF::runOnModule(Module &TheModule) {
  // Retrieve analyses
  auto &GCBI = getAnalysis<GeneratedCodeBasicInfoWrapperPass>().getGCBI();
  const auto &ModelWrapper = getAnalysis<LoadModelWrapperPass>().get();
  const model::Binary &Binary = *ModelWrapper.getReadOnlyModel();

  // Create an object of type IsolateFunctionsImpl and run the pass
  IFI Impl(TheModule.getFunction("root"), GCBI, Binary);
  Impl.run();

  return false;
}
