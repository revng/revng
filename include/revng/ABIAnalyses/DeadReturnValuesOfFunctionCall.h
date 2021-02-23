#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/ADT/DenseMap.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Value.h"

#include "revng/ABIAnalyses/DeadReturnValuesOfFunctionCall.h"
#include "revng/BasicAnalyses/GeneratedCodeBasicInfo.h"
#include "revng/MFP/MFP.h"
#include "revng/StackAnalysis/StackAnalysis.h"
#include "revng/Support/revng.h"

namespace DeadReturnValuesOfFunctionCall {

enum State { NoOrDead, Maybe, Unknown };
enum TransferKind { Read, Write, WeakWrite, TheCall, None };

// special register index, if found inside LatticeElement
// it means that it is the first time visiting the BasicBlock
const int32_t InitialRegisterState = -1;
struct MFI {
  using LatticeElement = llvm::DenseMap<int32_t, State>;
  using Label = const llvm::BasicBlock *;
  using GraphType = const llvm::BasicBlock *;

  MFI(const llvm::Instruction *CS,
      const llvm::DenseMap<llvm::GlobalVariable *, int32_t> &R,
      const StackAnalysis::FunctionProperties &P) :
    CallSite(CS), Properties(P), ABIRegisters(R){};

  const llvm::Instruction *CallSite;
  const StackAnalysis::FunctionProperties &Properties;
  const llvm::DenseMap<llvm::GlobalVariable *, int32_t> &ABIRegisters;
  bool isABIRegister(const llvm::Value *) const;
  TransferKind classifyInstruction(const llvm::Instruction *) const;
  llvm::SmallVector<int32_t, 1>
  getRegistersWritten(const llvm::Instruction *) const;
  llvm::SmallVector<int32_t, 1>
  getRegistersRead(const llvm::Instruction *) const;

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

namespace llvm {}; // namespace llvm
