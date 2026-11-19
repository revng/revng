#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/ADT/ArrayRef.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/DebugLoc.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"

#include "revng/Support/Assert.h"

/// Whether \p I is a `llvm.lifetime.start`.
inline bool isLifetimeStart(const llvm::Instruction &I) {
  const auto *Intrinsic = llvm::dyn_cast<llvm::IntrinsicInst>(&I);
  return Intrinsic != nullptr
         and Intrinsic->getIntrinsicID() == llvm::Intrinsic::lifetime_start;
}

/// Whether \p I is a `llvm.lifetime.end`.
inline bool isLifetimeEnd(const llvm::Instruction &I) {
  const auto *Intrinsic = llvm::dyn_cast<llvm::IntrinsicInst>(&I);
  return Intrinsic != nullptr
         and Intrinsic->getIntrinsicID() == llvm::Intrinsic::lifetime_end;
}

/// The `llvm.lifetime.start` marking where \p Alloca becomes live, or
/// `nullptr`.
///
/// There is at most one. `exit-ssa` emits one per variable it creates, and
/// `compute-alloca-lifetime-start` emits one only for the allocas that have
/// none; nothing in between duplicates it, because the passes that restructure
/// the CFG clone terminators rather than whole blocks.
inline const llvm::IntrinsicInst *
getLifetimeStart(const llvm::AllocaInst *Alloca) {
  const llvm::IntrinsicInst *Result = nullptr;

  for (const llvm::User *User : Alloca->users()) {
    const auto *I = llvm::dyn_cast<llvm::Instruction>(User);
    if (I == nullptr or not isLifetimeStart(*I))
      continue;

    revng_assert(Result == nullptr,
                 "a local variable has more than one lifetime.start");
    Result = llvm::cast<llvm::IntrinsicInst>(I);
  }

  return Result;
}

/// Whether \p Alloca carries a `llvm.lifetime.start`.
inline bool hasLifetimeStart(const llvm::AllocaInst *Alloca) {
  return getLifetimeStart(Alloca) != nullptr;
}

/// The variable a `llvm.lifetime.start` marks.
inline const llvm::AllocaInst *
getLifetimeStartVariable(const llvm::IntrinsicInst *LifetimeStart) {
  revng_assert(isLifetimeStart(*LifetimeStart));

  // The first argument is the size, the second the object
  const llvm::Value *Pointer = LifetimeStart->getArgOperand(1);
  return llvm::cast<llvm::AllocaInst>(Pointer->stripPointerCasts());
}

/// Emits the `llvm.lifetime.start` marking where \p Variable becomes live.
///
/// The marker goes before the nearest common dominator of every access to
/// \p Variable and of every instruction in \p AlsoLive, which is the latest
/// point whose result is available at all of them. Nothing is emitted, and
/// `nullptr` is returned, when there is no such point because the variable has
/// no accesses and \p AlsoLive is empty.
///
/// Placing the marker after an access would say the variable becomes live once
/// it has already been written. That makes the write dead as far as LLVM is
/// concerned, and leads `switch-to-statements` to read the marker as a write
/// of its own and copy the value it believes lost into a variable of its own.
///
/// \p AlsoLive carries the points the caller knows the variable has to be live
/// at although nothing accesses it there. `exit-ssa` passes the terminators of
/// the incoming blocks of the class of PHIs the variable stands for: an
/// undefined incoming produces no store, so without them the marker would sink
/// into whatever loop holds the accesses.
llvm::CallInst *emitLifetimeStart(llvm::AllocaInst *Variable,
                                  llvm::ArrayRef<llvm::Instruction *> AlsoLive,
                                  const llvm::DominatorTree &DT,
                                  const llvm::DebugLoc &Location);
