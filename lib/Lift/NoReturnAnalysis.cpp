//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/CFG.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"

#include "revng/ADT/Queue.h"
#include "revng/Lift/Helpers.h"
#include "revng/Model/Binary.h"
#include "revng/Model/FunctionTags.h"
#include "revng/Support/Debug.h"
#include "revng/Support/FunctionCallMarker.h"
#include "revng/Support/IRHelpers.h"

#include "NoReturnAnalysis.h"

using namespace llvm;

static Logger Log("noreturn-analysis");

using BlockSet = SmallPtrSet<BasicBlock *, 8>;

/// Collect the blocks transferring control to a dynamic function that the model
/// marks as `NoReturn`.
static BlockSet collectSeeds(Function &Root, const model::Binary &Binary) {
  BlockSet Seeds;

  // The marker is created by `TranslateDirectBranchesPass` out of what the
  // previous harvesting round materialized, so early on it does not exist yet
  std::optional JumpToSymbol = JumpToSymbolMarker.get(*Root.getParent());
  if (not JumpToSymbol.has_value())
    return Seeds;

  // TODO: restricting the seeds to imported symbols is arbitrary: local
  //       functions carry the same `NoReturn` attribute. We should add them,
  //       but we should also consider invalidation effects.
  const auto &DynamicFunctions = Binary.ImportedDynamicFunctions();
  for (IRHelperCall<NoArgument> Call : JumpToSymbol->callersIn(&Root)) {
    auto Name = extractFromConstantStringPtr(Call.call()->getArgOperand(0));
    if (Name.empty())
      continue;

    auto It = DynamicFunctions.find(Name.str());
    if (It == DynamicFunctions.end())
      continue;

    if (It->Attributes().contains(model::FunctionAttribute::NoReturn)) {
      BasicBlock *BB = Call.call()->getParent();
      revng_log(Log, getName(BB) << " jumps to noreturn symbol " << Name);
      Seeds.insert(BB);
    }
  }

  return Seeds;
}

static DenseMap<BasicBlock *, SmallVector<BasicBlock *, 2>>
buildCallGraph(Function &Root) {
  DenseMap<BasicBlock *, SmallVector<BasicBlock *, 2>> Callers;

  for (BasicBlock &BB : Root)
    if (BasicBlock *Callee = getFunctionCallCallee(&BB))
      Callers[Callee].push_back(&BB);

  return Callers;
}

SmallVector<BasicBlock *, 4>
cutNoReturnFallthroughs(Function &Root,
                        const model::Binary &Binary,
                        BasicBlock *UnknownTarget) {
  SmallVector<BasicBlock *, 4> DetachedFallthroughs;

  BlockSet Killers = collectSeeds(Root, Binary);
  if (Killers.empty()) {
    revng_log(Log, "No calls to noreturn symbols, nothing to do");
    return DetachedFallthroughs;
  }

  auto Callers = buildCallGraph(Root);

  // Point every killer at a single sink
  LLVMContext &Context = Root.getContext();
  auto *Sink = BasicBlock::Create(Context, "noreturn.sink", &Root);
  new UnreachableInst(Context, Sink);

  // A block jumping to a symbol already ends in `unreachable`, so there is no
  // successor to redirect: detach the terminator and branch to the sink
  // instead. Blocks that do have successors are redirected in place, which
  // keeps the `Use`s alive for whoever undoes our edits.
  using SuccessorList = SmallVector<BasicBlock *, 2>;
  SmallVector<std::pair<Instruction *, SuccessorList>> Redirected;
  SmallVector<std::pair<BasicBlock *, Instruction *>> Detached;

  for (BasicBlock *Killer : Killers) {
    Instruction *Terminator = Killer->getTerminator();

    if (Terminator->getNumSuccessors() == 0) {
      Terminator->removeFromParent();
      Detached.emplace_back(Killer, Terminator);
      BranchInst::Create(Sink, Killer);
    } else {
      Redirected.emplace_back(Terminator, SuccessorList(successors(Killer)));
      for (unsigned I = 0; I < Terminator->getNumSuccessors(); ++I)
        Terminator->setSuccessor(I, Sink);
    }
  }

  DominatorTreeBase<BasicBlock, /* IsPostDom = */ true> PostDominatorTree;
  PostDominatorTree.recalculate(Root);

  // Start from the sink and, every time a block turns out to be a killer, mark
  // as killers all its callers too: they cannot return either. Those in turn
  // bring in whatever they post-dominate.
  OnceQueue<BasicBlock *> WorkList;
  WorkList.insert(Sink);

  while (not WorkList.empty()) {
    BasicBlock *BB = WorkList.pop();

    SmallVector<BasicBlock *, 8> Descendants;
    PostDominatorTree.getDescendants(BB, Descendants);

    for (BasicBlock *NewKiller : Descendants) {
      if (NewKiller == Sink)
        continue;

      Killers.insert(NewKiller);

      auto It = Callers.find(NewKiller);
      if (It != Callers.end()) {
        for (BasicBlock *Caller : It->second) {
          if (Killers.insert(Caller).second) {
            revng_log(Log,
                      getName(Caller) << " calls " << getName(NewKiller)
                                      << ", which never returns");
            WorkList.insert(Caller);
          }
        }
      }
    }
  }

  // Restore the terminators we touched and drop the sink
  for (auto &[Terminator, Successors] : Redirected)
    for (unsigned I = 0; I < Successors.size(); ++I)
      Terminator->setSuccessor(I, Successors[I]);

  for (auto &[BB, Terminator] : Detached) {
    BB->getTerminator()->eraseFromParent();
    Terminator->insertInto(BB, BB->end());
  }

  Sink->eraseFromParent();

  // Detach the fallthrough of each call to a killer. Note that we cannot simply
  // detach the fallthrough of every killer: a killer can very well start with a
  // call to a function that *does* return (the one that never returns comes
  // later), and that fallthrough is genuinely reachable.
  for (BasicBlock &BB : Root) {
    BasicBlock *Callee = getFunctionCallCallee(&BB);
    if (Callee == nullptr or not Killers.contains(Callee))
      continue;

    Instruction *Terminator = BB.getTerminator();
    revng_assert(Terminator->getNumSuccessors() == 1);
    revng_log(Log,
              "Detaching the fallthrough of "
                << getName(&BB) << ": " << getName(Callee) << " never returns");
    DetachedFallthroughs.push_back(Terminator->getSuccessor(0));
    Terminator->setSuccessor(0, UnknownTarget);
  }

  revng_log(Log,
            "Detached " << DetachedFallthroughs.size() << " fallthrough edge(s)"
                        << ", " << Killers.size() << " killer basic block(s)");

  return DetachedFallthroughs;
}
