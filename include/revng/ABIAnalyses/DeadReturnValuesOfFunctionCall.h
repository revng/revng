#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/ADT/DenseMap.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Value.h"

#include "revng/ABIAnalyses/Common.h"
#include "revng/BasicAnalyses/GeneratedCodeBasicInfo.h"
#include "revng/MFP/MFP.h"
#include "revng/StackAnalysis/StackAnalysis.h"
#include "revng/Support/revng.h"

namespace DeadReturnValuesOfFunctionCall {

enum State { NoOrDead, Maybe, Unknown };

// special register index, if found inside LatticeElement
// it means that it is the first time visiting the BasicBlock
const int32_t InitialRegisterState = -1;
struct MFI : ABIAnalyses::ABIAnalysis {
  using LatticeElement = llvm::DenseMap<int32_t, State>;
  using Label = const llvm::BasicBlock *;
  using GraphType = const llvm::BasicBlock *;

  LatticeElement
  combineValues(const LatticeElement &Lh, const LatticeElement &Rh) const;
  LatticeElement applyTransferFunction(Label, const LatticeElement &E) const;
  bool isLessOrEqual(const LatticeElement &Lh, const LatticeElement &Rh) const;
};

llvm::DenseMap<llvm::GlobalVariable *, State>
analyze(const llvm::Instruction *CallSite,
        llvm::Function *Entry,
        const GeneratedCodeBasicInfo &GCBI,
        const StackAnalysis::FunctionProperties &FP);
} // namespace DeadReturnValuesOfFunctionCall
