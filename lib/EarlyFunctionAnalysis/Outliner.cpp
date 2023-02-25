/// \file Outliner.cpp

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/ADT/SCCIterator.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/Transforms/Utils/CodeExtractor.h"
#include "llvm/Transforms/Utils/ValueMapper.h"

#include "revng/ADT/Queue.h"
#include "revng/EarlyFunctionAnalysis/CallGraph.h"
#include "revng/EarlyFunctionAnalysis/CallHandler.h"
#include "revng/EarlyFunctionAnalysis/Outliner.h"
#include "revng/Model/IRHelpers.h"

using namespace llvm;

namespace efa {

class OutlinedFunctionsMap {
private:
  Module *M = nullptr;
  std::map<MetaAddress, UniqueValuePtr<Function>> Map;
  std::set<MetaAddress> Banned;

public:
  OutlinedFunctionsMap(Module *M) : M(M) {}
  ~OutlinedFunctionsMap() { clear(); }

public:
  auto begin() { return Map.begin(); }
  auto end() { return Map.end(); }

public:
  void clear() { Map.clear(); }

public:
  Function *get(MetaAddress Entry) {
    using namespace llvm;

    auto It = Map.find(Entry);
    if (It == Map.end()) {
      LLVMContext &Context = M->getContext();
      auto *FT = FunctionType::get(Type::getVoidTy(Context), {}, false);
      Function *New = Function::Create(FT,
                                       llvm::GlobalObject::ExternalLinkage,
                                       0,
                                       {},
                                       M);
      Map.insert(It, { Entry, UniqueValuePtr<Function>(New) });
      return New;
    } else {
      return It->second.get();
    }
  }

  void set(MetaAddress Entry, Function *F) { Map[Entry].reset(F); }

public:
  bool isBanned(MetaAddress BB) const { return Banned.contains(BB); }

  bool banRecursiveFunctions();
};

bool OutlinedFunctionsMap::banRecursiveFunctions() {
  using namespace llvm;
  bool Result = false;

  CallGraph CallGraph;
  std::map<Function *, BasicBlockNode *> ReverseMap;

  BasicBlockNode EntryNode(MetaAddress::invalid());
  BasicBlockNode *Entry = CallGraph.addNode(EntryNode);
  CallGraph.setEntryNode(Entry);

  // Populate graph and map
  for (auto &[Block, Function] : Map) {
    // We don't care about the address
    BasicBlockNode Node(MetaAddress::invalid());
    auto *NewNode = CallGraph.addNode(Node);
    ReverseMap[Function.get()] = NewNode;
    Entry->addSuccessor(NewNode);
  }

  // Add edges
  for (auto &[_, Function] : Map) {
    BasicBlockNode *Caller = ReverseMap.at(Function.get());
    for (Instruction &I : instructions(Function.get())) {
      if (auto *Call = dyn_cast<CallInst>(&I)) {
        auto It = ReverseMap.find(Call->getCalledFunction());
        if (It != ReverseMap.end()) {
          Caller->addSuccessor(It->second);
        }
      }
    }
  }

  // Ban everyone part of an SCC
  auto It = scc_begin(&CallGraph);
  auto End = scc_end(&CallGraph);
  for (; It != End; ++It) {
    if (not It.hasCycle())
      continue;

    auto &SCCNodes = *It;
    Result = Result or (SCCNodes.size() > 0);
    for (auto &Node : SCCNodes)
      Banned.insert(Node->Address);
  }

  return Result;
}

void Outliner::integrateFunctionCallee(CallHandler *TheCallHandler,
                                       MetaAddress CallerFunction,
                                       llvm::CallInst *FunctionCall,
                                       llvm::CallInst *JumpToSymbol,
                                       MetaAddress Callee,
                                       OutlinedFunctionsMap &FunctionsMap) {
  llvm::LLVMContext &Context = M.getContext();

  auto [Summary, IsTailCall] = getCallSiteInfo(TheCallHandler,
                                               CallerFunction,
                                               FunctionCall,
                                               JumpToSymbol,
                                               Callee);

  using namespace llvm;
  using llvm::BasicBlock;

  BasicBlock *BB = FunctionCall != nullptr ? FunctionCall->getParent() :
                                             JumpToSymbol->getParent();
  MetaAddress CallerBlock = getBasicBlockAddress(getJumpTargetBlock(BB));
  revng_assert(CallerBlock.isValid());

  bool TargetIsSymbol = JumpToSymbol != nullptr;
  Value *SymbolNamePointer = nullptr;
  if (TargetIsSymbol) {
    SymbolNamePointer = JumpToSymbol->getArgOperand(0);
  } else {
    using CPN = ConstantPointerNull;
    Type *I8Ptr = Type::getInt8PtrTy(Context);
    SymbolNamePointer = CPN::get(Type::getInt8PtrTy(Context));
  }

  if (FunctionCall == nullptr)
    IsTailCall = true;

  // Ignore any indirect tail call (i.e., not direct and not targeting a dynamic
  // symbol)
  if (IsTailCall and Callee.isInvalid() and SymbolNamePointer == nullptr)
    return;

  // The function call is replaced with 1) hooks that delimit the space of the
  // ABI analyses' traversals and 2) a summary of the registers clobbered by
  // that function.

  // What is the function type of the callee?
  using namespace model::FunctionAttribute;

  // Inline if 1) marked as inline, 2) not the caller function itself and 3)
  // not banned from inlining
  bool IsInline = false;
  bool IsCallToUnexpected = false;
  if (FunctionCall != nullptr) {
    auto *T = FunctionCall->getParent()->getTerminator();
    BasicBlock *CalleeBlock = getFunctionCallCallee(T);

    bool IsMarkedInline = Summary->Attributes.count(Inline) != 0;
    bool IsCallingTheCaller = CallerFunction == Callee;
    bool IsBanned = FunctionsMap.isBanned(Callee);
    IsInline = IsMarkedInline and not IsCallingTheCaller and not IsBanned;
  }

  bool IsNoReturn = Summary->Attributes.count(NoReturn) != 0;

  if (IsInline) {
    // Emit a call to the function to inline: a later step will inline this call
    revng_assert(Callee.isValid());
    auto *CI = CallInst::Create(FunctionsMap.get(Callee),
                                "",
                                BB->getTerminator());
  } else {
    Instruction *Term = BB->getTerminator();
    IRBuilder<> Builder(Term);
    TheCallHandler->handleCall(CallerBlock,
                               Builder,
                               Callee,
                               Summary->ClobberedRegisters,
                               Summary->ElectedFSO,
                               IsNoReturn,
                               IsTailCall,
                               SymbolNamePointer);

    if (IsNoReturn) {
      Term->eraseFromParent();
      Builder.SetInsertPoint(BB);
      TheCallHandler->handlePostNoReturn(Builder);
    }
  }

  // Erase markers
  if (FunctionCall != nullptr)
    eraseFromParent(FunctionCall);

  if (JumpToSymbol != nullptr)
    eraseFromParent(JumpToSymbol);
}

OutlinedFunction
Outliner::outlineFunctionInternal(CallHandler *TheCallHandler,
                                  llvm::BasicBlock *Entry,
                                  OutlinedFunctionsMap &FunctionsToInline) {
  using namespace llvm;
  using llvm::BasicBlock;

  MetaAddress FunctionAddress = getBasicBlockAddress(Entry);

  Function *Root = Entry->getParent();

  OutlinedFunction OutlinedFunction;
  OutlinedFunction.Address = FunctionAddress;
  OnceQueue<BasicBlock *> Queue;
  std::vector<BasicBlock *> BlocksToClone;

  auto *AnyPCBB = GCBI.anyPC();
  auto *UnexpectedPCBB = GCBI.unexpectedPC();

  Queue.insert(Entry);
  Queue.insert(AnyPCBB);
  Queue.insert(UnexpectedPCBB);

  //
  // Collect list of blocks to clone
  //
  while (not Queue.empty()) {
    BasicBlock *Current = Queue.pop();
    BlocksToClone.emplace_back(Current);

    if (auto *FunctionCall = getMarker(Current, "function_call")) {
      // Compute callee
      MetaAddress PCCallee = MetaAddress::invalid();
      if (auto *Next = getFunctionCallCallee(Current))
        PCCallee = getBasicBlockAddress(Next);

      CallInst *JumpToSymbol = getMarker(Current, "jump_to_symbol");
      auto [Summary, IsTailCall] = getCallSiteInfo(TheCallHandler,
                                                   FunctionAddress,
                                                   FunctionCall,
                                                   JumpToSymbol,
                                                   PCCallee);

      // Unless it's NoReturn, enqueue the call fallthrough
      using namespace model::FunctionAttribute;
      bool IsNoReturn = Summary->Attributes.count(NoReturn) != 0;
      if (not IsNoReturn and not IsTailCall) {
        Queue.insert(getFallthrough(Current));
      }
    } else {
      for (auto *Successor : successors(Current)) {
        if (not isPartOfRootDispatcher(Successor))
          Queue.insert(Successor);
      }
    }
  }

  //
  // Create a copy of all the basic blocks to outline in `root`
  //
  llvm::ValueToValueMapTy VMap;
  SmallVector<BasicBlock *, 8> BlocksToExtract;

  for (const auto &BB : BlocksToClone) {
    BasicBlock *Cloned = CloneBasicBlock(BB, VMap, Twine("_cloned"), Root);

    VMap[BB] = Cloned;
    BlocksToExtract.emplace_back(Cloned);
  }

  OutlinedFunction.AnyPCCloned = cast<BasicBlock>(VMap[AnyPCBB]);
  OutlinedFunction.UnexpectedPCCloned = cast<BasicBlock>(VMap[UnexpectedPCBB]);

  //
  // Update references in cloned basic blocks
  //
  remapInstructionsInBlocks(BlocksToExtract, VMap);

  //
  // Switch successor of function calls from callee to fallthrough
  //
  std::map<llvm::CallInst *, MetaAddress> CallToCallee;
  for (const auto &BB : BlocksToExtract) {
    if (CallInst *FunctionCall = getMarker(BB, "function_call")) {
      auto *Term = BB->getTerminator();

      // If the function callee is null, we are dealing with an indirect call
      MetaAddress PCCallee = MetaAddress::invalid();
      if (BasicBlock *Next = getFunctionCallCallee(Term))
        PCCallee = getBasicBlockAddress(Next);

      CallToCallee[FunctionCall] = PCCallee;

      using namespace model::FunctionAttribute;

      CallInst *JumpToSymbol = getMarker(BB, "jump_to_symbol");
      auto [CalleeSummary, IsTailCall] = getCallSiteInfo(TheCallHandler,
                                                         FunctionAddress,
                                                         FunctionCall,
                                                         JumpToSymbol,
                                                         PCCallee);

      bool IsNoReturn = CalleeSummary->Attributes.count(NoReturn) != 0;
      if (IsNoReturn) {
        auto *BB = Term->getParent();
        Term->eraseFromParent();
        IRBuilder<> Builder(BB);
        Builder.CreateUnreachable();

        // Ensure markers are still close to the terminator
        if (JumpToSymbol != nullptr)
          JumpToSymbol->moveBefore(BB->getTerminator());
        FunctionCall->moveBefore(BB->getTerminator());
      } else if (IsTailCall) {
        revng_assert(not cast<BranchInst>(Term)->isConditional());
        auto *Br = BranchInst::Create(cast<BasicBlock>(VMap[AnyPCBB]));
        ReplaceInstWithInst(Term, Br);
      } else {
        auto *Br = BranchInst::Create(getFallthrough(Term));
        ReplaceInstWithInst(Term, Br);
      }

      // To allow correct function extraction, there must not exist users of
      // BBs to be extracted, so we destroy the blockaddress of the
      // fall-through BB in the `function_call` marker.
      unsigned ArgNo = 0;
      PointerType *I8PtrTy = Type::getInt8PtrTy(M.getContext());
      Constant *I8NullPtr = ConstantPointerNull::get(I8PtrTy);
      for (Value *Arg : FunctionCall->args()) {
        if (isa<BlockAddress>(Arg)) {
          FunctionCall->setArgOperand(ArgNo, I8NullPtr);
          if (Arg->use_empty())
            cast<BlockAddress>(Arg)->destroyConstant();
        }
        ++ArgNo;
      }
    }
  }

  // Extract outlined function
  llvm::Function *F = CodeExtractor(BlocksToExtract).extractCodeRegion(CEAC);
  F->addFnAttr(Attribute::NullPointerIsValid);
  OutlinedFunction.Function = UniqueValuePtr<llvm::Function>(F);

  revng_assert(OutlinedFunction.Function != nullptr);
  revng_assert(OutlinedFunction.Function->arg_size() == 0);
  revng_assert(OutlinedFunction.Function->getReturnType()->isVoidTy());
  revng_assert(OutlinedFunction.Function->hasOneUser());

  // Remove the only user (call to the outlined function) in `root`
  auto It = OutlinedFunction.Function->user_begin();
  cast<Instruction>(*It)->getParent()->eraseFromParent();

  // Integrate function callee
  for (auto &BB : *OutlinedFunction.Function) {
    MetaAddress Callee;
    CallInst *FunctionCall = getMarker(&BB, "function_call");
    if (FunctionCall != nullptr)
      Callee = CallToCallee.at(FunctionCall);
    CallInst *JumpToSymbol = getMarker(&BB, "jump_to_symbol");
    // TODO: we don't integrate the call if it's a tail call (JumpToSymbol but
    //       no FunctionCall). Is this OK?
    if (FunctionCall != nullptr) {
      integrateFunctionCallee(TheCallHandler,
                              FunctionAddress,
                              FunctionCall,
                              JumpToSymbol,
                              Callee,
                              FunctionsToInline);
    }
  }

  return OutlinedFunction;
}

void Outliner::createAnyPCHooks(CallHandler *TheCallHandler,
                                OutlinedFunction *OutlinedFunction) {
  using namespace llvm;
  LLVMContext &Context = M.getContext();

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
    auto *JumpTargetBB = getJumpTargetBlock(BB);
    MetaAddress IndirectRetBBAddress = getBasicBlockAddress(JumpTargetBB);
    revng_assert(IndirectRetBBAddress.isValid());

    auto *Split = BB->splitBasicBlock(Term, BB->getName() + Twine("_anypc"));

    auto *JumpToAnyPC = Split->getTerminator();
    revng_assert(isa<BranchInst>(JumpToAnyPC));
    IRBuilder<> Builder(JumpToAnyPC);
    Builder.SetInsertPoint(JumpToAnyPC);

    using CPN = ConstantPointerNull;
    Value *SymbolName = CPN::get(Type::getInt8PtrTy(Context));
    FunctionSummary *Summary = nullptr;
    if (CallInst *JumpToSymbol = getMarker(BB, "jump_to_symbol")) {
      SymbolName = JumpToSymbol->getArgOperand(0);
      eraseFromParent(JumpToSymbol);
    }

    TheCallHandler->handleIndirectJump(Builder,
                                       IndirectRetBBAddress,
                                       SymbolName);
  }
}

class FixFunctionPreInlining {
private:
  unsigned InliningIndex = 0;
  std::set<std::pair<MetaAddress, llvm::Function *>> ToRestore;
  std::vector<MetaAddress> &InlinedFunctionsByIndex;

public:
  FixFunctionPreInlining(std::vector<MetaAddress> &InlinedFunctionsByIndex) :
    InlinedFunctionsByIndex(InlinedFunctionsByIndex) {}
  ~FixFunctionPreInlining() { revng_assert(ToRestore.size() == 0); }

public:
  void fix(const MetaAddress &CalleeAddress, llvm::CallBase *Caller) {
    Function *Callee = Caller->getCalledFunction();
    revng_assert(Callee != nullptr);
    setNewPCInlineIndex(CalleeAddress, Callee, ++InliningIndex);
    InlinedFunctionsByIndex.push_back(CalleeAddress);
    ToRestore.emplace(CalleeAddress, Callee);
  }

  void restore() {
    for (const auto &[Address, F] : ToRestore)
      setNewPCInlineIndex(Address, F, 0);
    ToRestore.clear();
  }

private:
  static void setNewPCInlineIndex(const MetaAddress &FunctionAddress,
                                  llvm::Function *F,
                                  unsigned InliningIndex) {
    revng_assert(FunctionAddress.isValid());

    for (llvm::Instruction &I : llvm::instructions(F)) {
      if (auto *NewPCCall = getCallTo(&I, "newpc")) {
        using namespace NewPCArguments;
        Value *Argument = NewPCCall->getArgOperand(InstructionID);
        auto OldID = BasicBlockID::fromValue(Argument);
        BasicBlockID NewID(OldID.start(), InliningIndex);
        NewPCCall->setArgOperand(InstructionID, NewID.toValue(F->getParent()));
      }
    }
  }
};

OutlinedFunction
Outliner::outline(llvm::BasicBlock *Entry, CallHandler *Handler) {
  using namespace llvm;

  OutlinedFunction Result;
  OutlinedFunctionsMap FunctionsToInline(&M);

  unsigned Attempts = 0;
  do {
    ++Attempts;
    revng_assert(Attempts < 3);

    // Restart isolating functions
    FunctionsToInline.clear();

    // Outline functions but do not perform inlining
    Result = outlineFunctionInternal(Handler, Entry, FunctionsToInline);

    //
    // Fixed point creation of function that needs to be inlined
    //
    bool SomethingNew = true;
    while (SomethingNew) {
      SomethingNew = false;
      for (const auto &[Address, F] : FunctionsToInline) {
        if (F->isDeclaration()) {
          Function *ToInline = createFunctionToInline(Handler,
                                                      GCBI.getBlockAt(Address),
                                                      FunctionsToInline);
          F->replaceAllUsesWith(ToInline);
          FunctionsToInline.set(Address, ToInline);
          SomethingNew = true;
        }
      }
    }

  } while (FunctionsToInline.banRecursiveFunctions());

  //
  // Fixed point inlining
  //
  bool SomethingNew = true;
  FixFunctionPreInlining FunctionFixer(Result.InlinedFunctionsByIndex);
  while (SomethingNew) {
    SomethingNew = false;
    for (auto &[Address, F] : FunctionsToInline) {
      SmallVector<CallBase *> ToInline;
      for (CallBase *Caller : callers(F.get()))
        ToInline.push_back(Caller);

      for (CallBase *Caller : ToInline) {
        FunctionFixer.fix(Address, Caller);
        InlineFunctionInfo IFI;
        bool Success = InlineFunction(*Caller, IFI).isSuccess();
        revng_assert(Success);
        SomethingNew = true;
      }
    }
  }

  FunctionFixer.restore();

  // Fix `unexpectedpc` of the callees to inline
  for (auto &I : instructions(Result.Function.get())) {
    if (CallInst *Call = getCallTo(&I, UnexpectedPCMarker.get())) {
      revng_assert(Result.UnexpectedPCCloned != nullptr);

      auto *Br = BranchInst::Create(Result.UnexpectedPCCloned);
      ReplaceInstWithInst(I.getParent()->getTerminator(), Br);
      Call->eraseFromParent();
      break;
    }
  }

  createAnyPCHooks(Handler, &Result);

  return Result;
}

llvm::Function *
Outliner::createFunctionToInline(CallHandler *TheCallHandler,
                                 llvm::BasicBlock *Entry,
                                 OutlinedFunctionsMap &FunctionsToInline) {
  using namespace llvm;
  LLVMContext &Context = M.getContext();

  // Recreate outlined function
  OutlinedFunction OF = outlineFunctionInternal(TheCallHandler,
                                                Entry,
                                                FunctionsToInline);

  // Adjust `anypc` and `unexpectedpc` BBs of the function to inline
  revng_assert(OF.AnyPCCloned != nullptr);

  // Functions to inline must have one and only one broken return
  revng_assert(OF.AnyPCCloned->hasNPredecessors(1));

  // Replace the broken return with a `ret`
  auto *Br = OF.AnyPCCloned->getUniquePredecessor()->getTerminator();
  auto *Ret = ReturnInst::Create(Context);
  ReplaceInstWithInst(Br, Ret);

  if (OF.UnexpectedPCCloned != nullptr) {
    CallInst::Create(UnexpectedPCMarker.get(),
                     "",
                     OF.UnexpectedPCCloned->getTerminator());
  }

  return OF.releaseFunction();
}

std::pair<const FunctionSummary *, bool>
Outliner::getCallSiteInfo(CallHandler *TheCallHandler,
                          MetaAddress CallerFunction,
                          llvm::CallInst *FunctionCall,
                          llvm::CallInst *JumpToSymbol,
                          MetaAddress Callee) {
  using namespace llvm;
  using llvm::BasicBlock;

  BasicBlock *BB = FunctionCall != nullptr ? FunctionCall->getParent() :
                                             JumpToSymbol->getParent();
  Instruction *Term = BB->getTerminator();

  // Extract MetaAddress of JT of the call-site
  BasicBlockID CallSiteAddress = getBasicBlockID(getJumpTargetBlock(BB));
  revng_assert(CallSiteAddress.isValid());

  StringRef CalledSymbol;
  if (JumpToSymbol != nullptr) {
    // The target is a symbol
    CalledSymbol = extractFromConstantStringPtr(JumpToSymbol->getArgOperand(0));
  }

  return Oracle.getCallSite(CallerFunction,
                            CallSiteAddress,
                            Callee,
                            CalledSymbol);
}

} // namespace efa
