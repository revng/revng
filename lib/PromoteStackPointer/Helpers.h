#pragma once

//
// Copyright (c) rev.ng Labs Srl. See LICENSE.md for details.
//

#include "revng/Support/MetaAddress.h"

inline MetaAddress getCallerBlockAddress(llvm::Instruction *I) {
  auto MaybeLocation = getLocation(I);
  revng_assert(MaybeLocation);
  return MaybeLocation->parent().back().start();
}

inline llvm::CallInst *findAssociatedCall(llvm::CallInst *SSACSCall) {
  using namespace llvm;

  // Look for the actual call in the same block or the next one
  Instruction *I = SSACSCall->getNextNode();
  while (I != SSACSCall) {
    if (auto *Call = getCallToIsolatedFunction(I)) {
      MetaAddress SSACSBlockAddress = getCallerBlockAddress(SSACSCall);
      revng_assert(getCallerBlockAddress(I) == SSACSBlockAddress);
      return Call;
    } else if (I->isTerminator()) {
      if (I->getNumSuccessors() != 1)
        return nullptr;
      I = I->getSuccessor(0)->getFirstNonPHI();
    } else {
      I = I->getNextNode();
    }
  }

  return nullptr;
}
