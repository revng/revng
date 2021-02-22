#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/ADT/DenseMap.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Value.h"

#include "revng/BasicAnalyses/GeneratedCodeBasicInfo.h"
#include "revng/MFP/MFP.h"
#include "revng/StackAnalysis/StackAnalysis.h"
#include "revng/Support/revng.h"

namespace DeadReturnValuesOfFunctionCall {

enum State { NoOrDead, Maybe, Unknown };
enum TransferKind { Read, Write, WeakWrite, TheCall, None };
struct MFI {
  using LatticeElement = llvm::DenseMap<int32_t, State>;
  using Label = const llvm::BasicBlock *;
  using GraphType = const llvm::Function *;

  MFI(const llvm::DenseMap<llvm::GlobalVariable *, int32_t> &R,
      StackAnalysis::FunctionProperties P) :
    Properties(P), ABIRegisters(R){};

  StackAnalysis::FunctionProperties Properties;
  llvm::DenseMap<llvm::GlobalVariable *, int32_t> ABIRegisters;
  bool isABIRegister(const llvm::Value *) const;
  TransferKind classifyInstruction(const llvm::Instruction &) const;
  llvm::SmallVector<int32_t, 1>
  getRegistersWritten(const llvm::Instruction &) const;
  llvm::SmallVector<int32_t, 1>
  getRegistersRead(const llvm::Instruction &) const;

  LatticeElement
  combineValues(const LatticeElement &Lh, const LatticeElement &Rh) const;
  LatticeElement applyTransferFunction(Label, const LatticeElement &E) const;
  bool isLessOrEqual(const LatticeElement &Lh, const LatticeElement &Rh) const;
};

inline std::map<MFI::Label, MFP::MFPResult<MFI::LatticeElement>>
analyze(llvm::Function *Entry, const GeneratedCodeBasicInfo &GCBI);
} // namespace DeadReturnValuesOfFunctionCall
