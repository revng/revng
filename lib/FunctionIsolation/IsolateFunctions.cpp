/// \file IsolateFunctions.cpp
/// \brief Implements the IsolateFunctions pass which applies function isolation
///        using the informations provided by FunctionBoundariesDetectionPass.

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/IR/IRBuilder.h"
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
#include "revng/FunctionIsolation/IsolateFunctions.h"
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

static void
eraseBranch(Instruction *I, BasicBlock *ExpectedUniqueSuccessor = nullptr) {
  auto *T = cast<BranchInst>(I);
  revng_assert(T->isUnconditional());
  if (ExpectedUniqueSuccessor != nullptr)
    revng_assert(T->getSuccessor(0) == ExpectedUniqueSuccessor);
  T->eraseFromParent();
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
  using BlockToFunctionsMap = std::map<BasicBlock *,
                                       std::pair<unsigned, Function *>>;
  using SuccessorsContainer = std::map<model::FunctionEdge, int>;

private:
  Function *RootFunction = nullptr;
  Module *TheModule = nullptr;
  LLVMContext &Context;
  GeneratedCodeBasicInfo &GCBI;
  const model::Binary &Binary;
  Function *RaiseException = nullptr;
  Function *FunctionDispatcher = nullptr;
  Function *CallMarker = nullptr;
  BlockToFunctionsMap IsolatedFunctionsMap;
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
  ///
  /// \return a pair of the entry block in root and the newly create Function
  std::pair<BasicBlock *, Function *> isolate(const model::Function &Function);

  /// Process a basic block from the model
  void handleBasicBlock(const model::BasicBlock &Block,
                        ValueToValueMapTy &OldToNew,
                        FunctionBlocks &ClonedBlocks);

  /// Clone the basic blocks involved in \p Entry jump target
  ///
  /// \return a vector of boundary basic blocks
  std::vector<Boundary>
  cloneAndIdentifyBoundaries(MetaAddress Entry,
                             ValueToValueMapTy &OldToNew,
                             FunctionBlocks &ClonedBlocks);

  /// Create the code necessary to handle a direct branch in the IR
  bool handleDirectBoundary(const Boundary &TheBoundary,
                            SuccessorsContainer &ExpectedSuccessors,
                            FunctionBlocks &ClonedBlocks);

  /// Create the code necessary to handle an indirect branch in the IR
  bool handleIndirectBoundary(const std::vector<Boundary> &Boundaries,
                              const model::BasicBlock &Block,
                              const SuccessorsContainer &ExpectedSuccessors,
                              bool CallConsumed,
                              FunctionBlocks &ClonedBlocks);

  /// Emit a function call marker and a branch to the return address
  void createFunctionCall(IRBuilder<> &Builder,
                          MetaAddress ExpectedCallee,
                          const Boundary &TheBoundary,
                          FunctionBlocks &ClonedBlocks);

  void createFunctionCall(BasicBlock *BB,
                          MetaAddress Callee,
                          const Boundary &TheBoundary,
                          FunctionBlocks &ClonedBlocks) {
    IRBuilder<> Builder(BB);
    createFunctionCall(Builder, Callee, TheBoundary, ClonedBlocks);
  }

  /// Post process all the call markers, replacing them with actual calls
  void replaceCallMarker() const;

  /// Populate the function_dispatcher, needed to handle the indirect calls
  void populateFunctionDispatcher();

  /// Create code to throw of an exception
  void throwException(IRBuilder<> &Builder,
                      StringRef Reason,
                      const DebugLoc &DbgLocation);

  void throwException(BasicBlock *BB,
                      StringRef Reason,
                      const DebugLoc &DbgLocation) {
    IRBuilder<> Builder(BB);
    throwException(Builder, Reason, DbgLocation);
  }
};

void IFI::throwException(IRBuilder<> &Builder,
                         StringRef Reason,
                         const DebugLoc &DbgLocation) {
  revng_assert(RaiseException != nullptr);
  revng_assert(DbgLocation);

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
  revng_assert(Dbg);
  throwException(Unexpected, "An unexpected functions has been called", Dbg);
  setBlockType(Unexpected->getTerminator(), BlockType::UnexpectedPCBlock);

  IRBuilder<> Builder(Context);

  // Create all the entries of the dispatcher
  ProgramCounterHandler::DispatcherTargets Targets;
  for (auto &[Block, P] : IsolatedFunctionsMap) {
    auto &[_, F] = P;

    BasicBlock *Trampoline = BasicBlock::Create(Context,
                                                F->getName() + "_trampoline",
                                                FunctionDispatcher,
                                                nullptr);
    Targets.emplace_back(GCBI.getPCFromNewPC(&*Block->begin()), Trampoline);

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
static bool any(const T &Range, const F &Predicate) {
  auto End = Range.end();
  return End != std::find_if(Range.begin(), End, Predicate);
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

static bool isDirectEdge(model::FunctionEdgeType::Values Type) {
  using namespace model::FunctionEdgeType;

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
    revng_abort();
  }
}

static bool isIndirectEdge(model::FunctionEdgeType::Values Type) {
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
                                 const model::BasicBlock &Block,
                                 const SuccessorsContainer &ExpectedSuccessors,
                                 bool CallConsumed,
                                 FunctionBlocks &ClonedBlocks) {
  std::vector<model::FunctionEdge> RemainingEdges;
  for (auto &[Edge, EdgeUsageCount] : ExpectedSuccessors)
    if (EdgeUsageCount == 0)
      RemainingEdges.push_back(Edge);

  // At this point RemainingEdges must either be a series of
  // `DirectBranch` or a single one of the indirect ones
  bool NoMore = RemainingEdges.size() == 0;
  using namespace model::FunctionEdgeType;

  auto IsDirectEdge = [](const model::FunctionEdge &E) {
    return isDirectEdge(E.Type);
  };
  bool AllDirect = allOrNone(RemainingEdges, IsDirectEdge, false);

  auto IndirectType = Invalid;
  if (not NoMore and not AllDirect) {
    // We're not out of expected successors, but they are not all direct.
    // We expect a single indirect edge.
    revng_assert(RemainingEdges.size() == 1);
    IndirectType = RemainingEdges.begin()->Type;
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
    revng_assert(NoMore);
    return false;
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
      for (const model::FunctionEdge &Edge : RemainingEdges)
        Inserter.insert(Edge.Destination);
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
                         ClonedBlocks);
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

  OldTerminator->eraseFromParent();

  return true;
}

/// \return true if this was a call
bool IFI::handleDirectBoundary(const Boundary &TheBoundary,
                               SuccessorsContainer &ExpectedSuccessors,
                               FunctionBlocks &ClonedBlocks) {
  BasicBlock *BB = TheBoundary.Block;
  SetAtMostOnce IsCall;
  size_t Consumed = 0;
  for (auto &[Edge, EdgeUsageCount] : ExpectedSuccessors) {
    // Is this edge targeting who we expect?
    if (TheBoundary.Successors.Addresses.count(Edge.Destination) == 0)
      continue;

    bool Match = false;
    switch (Edge.Type) {
    case model::FunctionEdgeType::DirectBranch:
      if (not TheBoundary.isCall())
        Match = true;
      break;

    case model::FunctionEdgeType::FunctionCall:
    case model::FunctionEdgeType::FakeFunctionCall:
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

      if (Edge.Type == model::FunctionEdgeType::FunctionCall) {
        eraseBranch(BB->getTerminator(), TheBoundary.CalleeBlock);
        createFunctionCall(BB, Edge.Destination, TheBoundary, ClonedBlocks);
      }
    }
  }

  revng_assert(Consumed == TheBoundary.Successors.Addresses.size());

  return IsCall;
}

std::vector<Boundary>
IFI::cloneAndIdentifyBoundaries(MetaAddress Entry,
                                ValueToValueMapTy &OldToNew,
                                FunctionBlocks &ClonedBlocks) {
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
    if (succ_empty(BB) or any(successors(BB), HasNotBeenCloned)) {
      BasicBlock *Callee = getFunctionCallCallee(BB);
      BasicBlock *Fallthrough = getFallthrough(BB);
      Boundary NewBoundary{
        NewBB, Callee, Fallthrough, GCBI.getSuccessors(BB)
      };
      Boundaries.push_back(NewBoundary);
    }
  }

  return Boundaries;
}

void IFI::handleBasicBlock(const model::BasicBlock &Block,
                           ValueToValueMapTy &OldToNew,
                           FunctionBlocks &ClonedBlocks) {
  // Sentinel to ensure we don't have more than a call within a basic block
  SetAtMostOnce CallConsumed;

  // Identify boundary blocks
  std::vector<Boundary> Boundaries = cloneAndIdentifyBoundaries(Block.Start,
                                                                OldToNew,
                                                                ClonedBlocks);

  // At this point, we first need to handle all the boundary blocks that
  // represent direct jumps, then we'll take care of the (only) indirect jump,
  // if any

  SuccessorsContainer ExpectedSuccessors;
  for (const auto &E : Block.Successors) {
    // Ignore self-loops
    if (E->Destination != Block.Start) {
      ExpectedSuccessors[*E] = 0;
    }
  }

  int IndirectCount = 0;

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

  // Consume direct jumps calls
  for (const auto &Boundary : Boundaries) {
    if (not(Boundary.Successors.AnyPC or Boundary.Successors.UnexpectedPC)) {
      bool Result = handleDirectBoundary(Boundary,
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

std::pair<BasicBlock *, Function *>
IFI::isolate(const model::Function &Function) {
  // Map from origina values to new ones
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
  revng_assert(Dbg);
  throwException(ClonedBlocks.unexpectedPCBlock(), "unexpectedPC", Dbg);
  OldToNew[GCBI.unexpectedPC()] = ClonedBlocks.unexpectedPCBlock();

  // Get the entry basic block
  BasicBlock *OriginalEntry = GCBI.getBlockAt(Function.Entry);

  TheLogger << "Isolating ";
  Function.Entry.dump(TheLogger);
  TheLogger << DoLog;
  LoggerIndent<> Indent(TheLogger);

  for (const model::BasicBlock &Block : Function.CFG) {

    if (TheLogger.isEnabled()) {
      TheLogger << "Isolating ";
      Block.Start.dump(TheLogger);
      TheLogger << "-";
      Block.End.dump(TheLogger);
      TheLogger << DoLog;
    }

    LoggerIndent<> Indent2(TheLogger);

    // Process the basic block
    handleBasicBlock(Block, OldToNew, ClonedBlocks);
  }

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
    I->eraseFromParent();

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
  FunctionTags::Lifted.addTo(NewFunction);

  revng_assert(NewFunction != nullptr);
  NewFunction->setName(Function.name());

  FunctionType *FT = NewFunction->getFunctionType();
  revng_assert(FT->getReturnType()->isVoidTy());
  revng_assert(FT->getNumParams() == 0);

  return { OriginalEntry, NewFunction };
}

void IFI::createFunctionCall(IRBuilder<> &Builder,
                             MetaAddress ExpectedCallee,
                             const Boundary &TheBoundary,
                             FunctionBlocks &ClonedBlocks) {
  BasicBlock *ExpectedCalleeBB = nullptr;
  unsigned CalleeIndex = 0;
  if (ExpectedCallee.isValid()) {
    ExpectedCalleeBB = GCBI.getBlockAt(ExpectedCallee);
    CalleeIndex = IsolatedFunctionsMap.at(ExpectedCalleeBB).first;
  }

  if (TheBoundary.CalleeBlock != nullptr
      and TheBoundary.CalleeBlock != ExpectedCalleeBB) {
    revng_log(TheLogger,
              "Warning: The callee in the binary ("
                << getName(TheBoundary.CalleeBlock)
                << ") is different from the one provided by the model ("
                << getName(ExpectedCalleeBB) << ")");
  }

  BasicBlock::iterator InsertPoint = Builder.GetInsertPoint();
  revng_assert(not Builder.GetInsertBlock()->empty());
  Instruction *Old = InsertPoint == Builder.GetInsertBlock()->end() ?
                       &*Builder.GetInsertBlock()->rbegin() :
                       &*InsertPoint;
  auto *NewCall = Builder.CreateCall(CallMarker, Builder.getInt32(CalleeIndex));
  NewCall->setDebugLoc(Old->getDebugLoc());

  if (TheBoundary.ReturnBlock != nullptr) {
    // Emit jump to fallthrough
    Builder.CreateBr(TheBoundary.ReturnBlock);
  } else {
    if (TheLogger.isEnabled()) {
      TheLogger << "Call to ";
      ExpectedCallee.dump(TheLogger);
      TheLogger << " in " << getName(TheBoundary.Block)
                << " has not been detected as a function call in the binary."
                << DoLog;
    }

    throwException(Builder,
                   "An instruction marked as a call has not been "
                   "identified as such in the binary",
                   Old->getDebugLoc());
  }
}

void IFI::replaceCallMarker() const {
  std::vector<Function *> Functions;
  Functions.resize(IsolatedFunctionsMap.size() + 1);

  Functions[0] = FunctionDispatcher;
  for (auto [_, P] : IsolatedFunctionsMap) {
    auto [Index, Function] = P;
    revng_assert(Index != 0);
    revng_assert(Function != nullptr);
    Functions[Index] = Function;
  }

  for (auto It = CallMarker->user_begin(); It != CallMarker->user_end();) {
    User *U = *It;
    ++It;

    auto *Call = cast<CallInst>(U);
    Value *CallMarkerArgument = Call->getArgOperand(0);
    unsigned Index = cast<ConstantInt>(CallMarkerArgument)->getLimitedValue();
    Function *Callee = Functions[Index];
    auto *NewCall = CallInst::Create(FunctionCallee{ Callee }, "", Call);
    NewCall->setDebugLoc(Call->getDebugLoc());
    Call->eraseFromParent();
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

  auto *CallMarkerFTy = createFunctionType<void, uint32_t>(Context);
  CallMarker = Function::Create(CallMarkerFTy,
                                GlobalValue::ExternalLinkage,
                                "call_marker",
                                TheModule);

  unsigned I = 1;
  for (const model::Function &Function : Binary.Functions) {
    if (Function.Type == model::FunctionType::Fake)
      continue;
    IsolatedFunctionsMap[GCBI.getBlockAt(Function.Entry)].first = I;
    ++I;
  }

  std::set<Function *> IsolatedFunctions;
  for (const model::Function &Function : Binary.Functions) {
    // Do not isolate fake functions
    if (Function.Type == model::FunctionType::Fake)
      continue;

    // Perform isolation
    auto [EntryBlock, IsolatedFunction] = isolate(Function);

    // Record new isolated function
    BasicBlock *OriginalEntry = GCBI.getBlockAt(Function.Entry);
    IsolatedFunctions.insert(IsolatedFunction);
    IsolatedFunctionsMap.at(OriginalEntry).second = IsolatedFunction;
  }

  replaceCallMarker();

  this->CallMarker->eraseFromParent();
  this->CallMarker = nullptr;

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
}

bool IF::runOnModule(Module &TheModule) {
  // Retrieve analyses
  auto &GCBI = getAnalysis<GeneratedCodeBasicInfoWrapperPass>().getGCBI();
  const auto &ModelWrapper = getAnalysis<LoadModelWrapperPass>().get();
  const model::Binary &Binary = ModelWrapper.getReadOnlyModel();

  // Create an object of type IsolateFunctionsImpl and run the pass
  IFI Impl(TheModule.getFunction("root"), GCBI, Binary);
  Impl.run();

  return false;
}
