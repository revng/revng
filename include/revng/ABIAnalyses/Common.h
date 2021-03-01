#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/ADT/DenseMap.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Value.h"

#include "revng/BasicAnalyses/GeneratedCodeBasicInfo.h"
#include "revng/MFP/MFP.h"
#include "revng/Model/Binary.h"
#include "revng/StackAnalysis/StackAnalysis.h"
#include "revng/Support/revng.h"

namespace ABIAnalyses {
using Register = model::Register::Values;

enum TransferKind {
  Read,
  Write,
  WeakWrite,
  TheCall,
  None,
  // legacy transfer functions
  ReturnFromMaybe,
  ReturnFromNoOrDead,
  ReturnFromUnknown,
  UnknownFunctionCall
};

struct ABIAnalysis {

  ABIAnalysis(const GeneratedCodeBasicInfo &GCBI,
              const StackAnalysis::FunctionProperties &P) :
    ABIAnalysis(nullptr, GCBI, P){};

  ABIAnalysis(const llvm::Instruction *CS,
              const GeneratedCodeBasicInfo &GCBI,
              const StackAnalysis::FunctionProperties &P) :
    CallSite(CS), Properties(P), GCBI(GCBI) {

    for (auto *CSV : GCBI.abiRegisters())
      if (CSV) {
        Register Register = model::Register::fromRegisterName(CSV->getName(),
                                                              GCBI.arch());
        if (Register != model::Register::Invalid) {
          ABIRegisters[CSV] = Register;
          IndexABIRegisters[ABIRegisters[CSV]] = CSV;
        }
      }
  };

  llvm::DenseMap<llvm::GlobalVariable *, Register> ABIRegisters;
  llvm::DenseMap<Register, llvm::GlobalVariable *> IndexABIRegisters;

  const llvm::Instruction *CallSite;
  const StackAnalysis::FunctionProperties &Properties;
  const GeneratedCodeBasicInfo &GCBI;

  bool isABIRegister(const llvm::Value *) const;
  bool isABIRegister(Register RegID) const;

  /// Returns the corresponding global variable for a RegID or nullptr if the id
  /// is invalid
  llvm::GlobalVariable *getABIRegister(Register RegID) const;

  TransferKind classifyInstruction(const llvm::Instruction *) const;

  llvm::SmallVector<Register, 1>
  getRegistersWritten(const llvm::Instruction *) const;

  llvm::SmallVector<Register, 1>
  getRegistersRead(const llvm::Instruction *) const;
};

inline bool ABIAnalysis::isABIRegister(const llvm::Value *V) const {
  if (const auto &G = llvm::dyn_cast<llvm::GlobalVariable>(V)) {
    if (ABIRegisters.count(G) != 0) {
      return true;
    }
  }
  return false;
}

inline bool ABIAnalysis::isABIRegister(Register RegID) const {
  return IndexABIRegisters.count(RegID);
}

inline llvm::GlobalVariable *ABIAnalysis::getABIRegister(Register RegID) const {
  if (isABIRegister(RegID)) {
    return IndexABIRegisters.lookup(RegID);
  }
  return nullptr;
}

inline TransferKind
ABIAnalysis::classifyInstruction(const llvm::Instruction *I) const {
  switch (I->getOpcode()) {
  case llvm::Instruction::Store: {
    auto S = llvm::cast<llvm::StoreInst>(I);
    if (isABIRegister(S->getPointerOperand())) {
      return Write;
    }
    break;
  }
  case llvm::Instruction::Load: {
    auto L = llvm::cast<llvm::LoadInst>(I);
    if (isABIRegister(L->getPointerOperand())) {
      return Read;
    }
    break;
  }
  case llvm::Instruction::Call: {
    if (I == CallSite) {
      return TheCall;
    }
    return WeakWrite;
  }
  }
  return None;
}

inline llvm::SmallVector<Register, 1>
ABIAnalysis::getRegistersWritten(const llvm::Instruction *I) const {
  llvm::SmallVector<Register, 1> Result;
  switch (I->getOpcode()) {
  case llvm::Instruction::Store: {
    auto S = llvm::cast<llvm::StoreInst>(I);
    if (isABIRegister(S->getPointerOperand())) {
      Result.push_back(ABIRegisters.lookup(
        llvm::cast<llvm::GlobalVariable>(S->getPointerOperand())));
    }
    break;
  }
  }
  return Result;
}

inline llvm::SmallVector<Register, 1>
ABIAnalysis::getRegistersRead(const llvm::Instruction *I) const {
  llvm::SmallVector<Register, 1> Result;
  switch (I->getOpcode()) {
  case llvm::Instruction::Load: {
    auto L = llvm::cast<llvm::LoadInst>(I);
    if (isABIRegister(L->getPointerOperand())) {
      Result.push_back(ABIRegisters.lookup(
        llvm::cast<llvm::GlobalVariable>(L->getPointerOperand())));
    }
    break;
  }
  }
  return Result;
}
} // namespace ABIAnalyses
