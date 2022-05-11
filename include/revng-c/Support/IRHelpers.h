#pragma once

//
// Copyright (c) rev.ng Labs Srl. See LICENSE.md for details.
//

#include <optional>

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/GlobalValue.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Type.h"

#include "revng/Support/Assert.h"
#include "revng/Support/Debug.h"
#include "revng/Support/IRHelpers.h"

inline void
pushInstructionALAP(llvm::DominatorTree &DT, llvm::Instruction *ToMove) {
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
        CommonDominator = DT.findNearestCommonDominator(CommonDominator, BB);
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

namespace llvm {

class InsertValueInst;
class Instruction;
class ExtractValueInst;
class Value;

} // end namespace llvm

extern llvm::SmallVector<llvm::Value *, 2>
getInsertValueLeafOperands(llvm::InsertValueInst *);

extern llvm::SmallVector<const llvm::Value *, 2>
getInsertValueLeafOperands(const llvm::InsertValueInst *);

extern llvm::SmallVector<llvm::SmallPtrSet<llvm::ExtractValueInst *, 2>, 2>
getExtractedValuesFromInstruction(llvm::Instruction *);

extern llvm::SmallVector<llvm::SmallPtrSet<const llvm::ExtractValueInst *, 2>,
                         2>
getExtractedValuesFromInstruction(const llvm::Instruction *);

/// Deletes the body of an llvm::Function, but preservin all the tags and
/// attributes (which llvm::Function::deleteBody() does not preserve).
/// Returns true if the body was cleared, false if it was already empty.
extern bool deleteOnlyBody(llvm::Function &F);
