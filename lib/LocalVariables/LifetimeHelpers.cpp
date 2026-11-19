//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Instructions.h"

#include "revng/LocalVariables/LifetimeHelpers.h"
#include "revng/Support/IRBuilder.h"

llvm::CallInst *emitLifetimeStart(llvm::AllocaInst *Variable,
                                  llvm::ArrayRef<llvm::Instruction *> AlsoLive,
                                  const llvm::DominatorTree &DT,
                                  const llvm::DebugLoc &Location) {
  llvm::Instruction *InsertBefore = nullptr;

  auto Join = [&DT, &InsertBefore](llvm::Instruction *I) {
    // An unreachable point has no dominance relation to go by, and joining one
    // would make the result depend on the order the points come in.
    revng_assert(DT.isReachableFromEntry(I->getParent()),
                 "a lifetime.start cannot be placed from unreachable code");

    InsertBefore = InsertBefore == nullptr ?
                     I :
                     DT.findNearestCommonDominator(InsertBefore, I);
  };

  // The use list comes in whatever order it comes in. The nearest common
  // dominator of a set of instructions does not depend on the order they are
  // joined in.
  for (llvm::User *TheUser : Variable->users())
    Join(llvm::cast<llvm::Instruction>(TheUser));

  for (llvm::Instruction *I : AlsoLive)
    Join(I);

  // Nothing accesses the variable, and the caller knows of no point where it
  // has to be live: there is nothing to mark.
  if (InsertBefore == nullptr)
    return nullptr;

  revng::IRBuilder Builder(InsertBefore, Location);

  // The size is left unspecified, which stands for the whole object. The
  // marker is only here to carry the declaration point, and nothing downstream
  // reasons about how much of the object is live.
  return Builder.CreateLifetimeStart(Variable);
}
