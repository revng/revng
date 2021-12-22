#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <optional>

#include "llvm/IR/Constants.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/GlobalValue.h"
#include "llvm/IR/Instructions.h"

#include "revng/Support/Assert.h"
#include "revng/Support/Debug.h"
#include "revng/Support/IRHelpers.h"

inline void pushALAP(llvm::DominatorTree &DT, llvm::Instruction *ToMove) {
  using namespace llvm;

  std::set<Instruction *> Users;
  BasicBlock *CommonDominator = nullptr;
  for (User *U : ToMove->users()) {
    if (auto *I = dyn_cast<Instruction>(U)) {
      Users.insert(I);
      auto *BB = I->getParent();
      if (CommonDominator == nullptr) {
        CommonDominator = BB;
      } else {
        DT.findNearestCommonDominator(CommonDominator, BB);
      }
    }
  }

  revng_assert(CommonDominator != nullptr);

  for (Instruction &I : *CommonDominator) {
    if (I.isTerminator() or Users.count(&I) != 0) {
      ToMove->moveBefore(&I);
      return;
    }
  }

  revng_abort("Block has no terminator");
}

template<typename T>
inline std::optional<T> getConstantArg(llvm::CallInst *Call, unsigned Index) {
  using namespace llvm;

  if (auto *CI = dyn_cast<ConstantInt>(Call->getArgOperand(Index))) {
    return CI->getLimitedValue();
  } else {
    return {};
  }
}

inline std::optional<uint64_t>
getUnsignedConstantArg(llvm::CallInst *Call, unsigned Index) {
  return getConstantArg<uint64_t>(Call, Index);
}

inline std::optional<int64_t>
getSignedConstantArg(llvm::CallInst *Call, unsigned Index) {
  return getConstantArg<int64_t>(Call, Index);
}
