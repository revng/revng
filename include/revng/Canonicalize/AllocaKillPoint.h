#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Analysis/PostDominators.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Value.h"
#include "llvm/Support/Casting.h"

/// Returns the block past which the value held by \p Alloca is dead, i.e. no
/// instruction that transitively depends on that value is reachable on any path
/// from there on. This is the nearest common post-dominator of the whole
/// forward use closure of \p Alloca: its direct users, their users, and so on
/// (following loaded values into whatever consumes them). It is nullptr when
/// that closure has no common post-dominator (e.g. because of multiple function
/// exits) and when \p Alloca has no uses.
///
/// The closure, rather than just the direct uses, is what matters: a value
/// loaded from the alloca stays live at the instructions that consume the
/// loaded value, which live past the load's own block.
inline const llvm::BasicBlock *
commonPostDominatorOfTransitiveUses(const llvm::PostDominatorTree &PDT,
                                    const llvm::AllocaInst *Alloca) {
  using llvm::BasicBlock;
  using llvm::Instruction;
  using llvm::User;

  // Forward-closure worklist over the use graph, seeded with the alloca's
  // direct users. The visited set avoids repeated work and breaks the cycles
  // that PHI nodes introduce in loops. The closure naturally terminates at
  // void-typed users (stores, returns), which have no users of their own.
  llvm::SmallPtrSet<const Instruction *, 16> Visited;
  llvm::SmallVector<const Instruction *, 16> Worklist;
  auto Enqueue = [&Visited, &Worklist](const llvm::Value *V) {
    for (const User *U : V->users())
      if (const auto *I = llvm::dyn_cast<Instruction>(U))
        if (Visited.insert(I).second)
          Worklist.push_back(I);
  };
  Enqueue(Alloca);

  const BasicBlock *CommonPostDom = nullptr;
  while (not Worklist.empty()) {
    const Instruction *I = Worklist.pop_back_val();

    const BasicBlock *UseBlock = I->getParent();
    if (CommonPostDom == nullptr)
      CommonPostDom = UseBlock;
    else
      CommonPostDom = PDT.findNearestCommonDominator(CommonPostDom, UseBlock);

    // No common post-dominator at all: give up.
    if (CommonPostDom == nullptr)
      break;

    Enqueue(I);
  }

  return CommonPostDom;
}

/// Returns true when \p I lies past \p CommonPostDom, i.e. \p CommonPostDom no
/// longer post-dominates the block of \p I and is therefore strictly upstream
/// of \p I. Always false when \p CommonPostDom is nullptr.
inline bool isPastCommonPostDominator(const llvm::PostDominatorTree &PDT,
                                      const llvm::BasicBlock *CommonPostDom,
                                      const llvm::Instruction *I) {
  if (CommonPostDom == nullptr)
    return false;
  return not PDT.dominates(CommonPostDom, I->getParent());
}
