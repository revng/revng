#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/IR/Instructions.h"

#include "revng/Support/Debug.h"
#include "revng/Support/IRHelpers.h"

namespace llvm {
class APInt;
}

inline Logger<> ValueMaterializerLogger("value-materializer");

std::string aviFormatter(const llvm::APInt &Value);

inline bool isPhiLike(llvm::Value *V) {
  return (llvm::isa<llvm::PHINode>(V) or llvm::isa<llvm::SelectInst>(V));
}

inline bool isMemory(llvm::Value *V) {
  using namespace llvm;
  V = skipCasts(V);
  return not(isa<GlobalVariable>(V) or isa<AllocaInst>(V));
}
