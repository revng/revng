#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Value.h"

#include "revng/MFP/MFP.h"
#include "revng/Support/revng.h"

namespace DeadReturnValuesOfFunctionCall {

enum State { Unknown, Maybe, NoOrDead };

struct MFI {
  using LatticeElement = std::map<const llvm::GlobalVariable *, State>;
  using Label = llvm::BasicBlock *;
  using GraphType = llvm::Function *;
  static LatticeElement
  combineValues(const LatticeElement &Lh, const LatticeElement &Rh);
  static LatticeElement applyTransferFunction(Label, const LatticeElement &E);
  static bool isLessOrEqual(const LatticeElement &Lh, const LatticeElement &Rh);
};

static void analyze(llvm::BasicBlock *Entry);
} // namespace DeadReturnValuesOfFunctionCall
