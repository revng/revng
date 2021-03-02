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
using namespace llvm;
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
private:

  DenseMap<GlobalVariable *, Register> ABIRegisters;
  DenseMap<Register, GlobalVariable *> IndexABIRegisters;

  const Instruction *CallSite;

public:
  ABIAnalysis(const GeneratedCodeBasicInfo &GCBI) :
    ABIAnalysis(nullptr, GCBI){};

  ABIAnalysis(const Instruction *CS,
              const GeneratedCodeBasicInfo &GCBI) :
    CallSite(CS) {

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


  bool isABIRegister(const Value *) const;
  bool isABIRegister(Register RegID) const;

  /// Returns the corresponding global variable for a RegID or nullptr if the id
  /// is invalid
  GlobalVariable *getABIRegister(Register RegID) const;

  TransferKind classifyInstruction(const Instruction *) const;

  SmallVector<Register, 1>
  getRegistersWritten(const Instruction *) const;

  SmallVector<Register, 1>
  getRegistersRead(const Instruction *) const;
};

inline bool ABIAnalysis::isABIRegister(const Value *V) const {
  if (const auto &G = dyn_cast<GlobalVariable>(V)) {
    if (ABIRegisters.count(G) != 0) {
      return true;
    }
  }
  return false;
}

inline bool ABIAnalysis::isABIRegister(Register RegID) const {
  return IndexABIRegisters.count(RegID);
}

inline GlobalVariable *ABIAnalysis::getABIRegister(Register RegID) const {
  if (isABIRegister(RegID)) {
    return IndexABIRegisters.lookup(RegID);
  }
  return nullptr;
}

inline TransferKind
ABIAnalysis::classifyInstruction(const Instruction *I) const {
  switch (I->getOpcode()) {
  case Instruction::Store: {
    auto S = cast<StoreInst>(I);
    if (isABIRegister(S->getPointerOperand())) {
      return Write;
    }
    break;
  }
  case Instruction::Load: {
    auto L = cast<LoadInst>(I);
    if (isABIRegister(L->getPointerOperand())) {
      return Read;
    }
    break;
  }
  case Instruction::Call: {
    if (I == CallSite) {
      return TheCall;
    }
    return WeakWrite;
  }
  }
  return None;
}

inline SmallVector<Register, 1>
ABIAnalysis::getRegistersWritten(const Instruction *I) const {
  SmallVector<Register, 1> Result;
  switch (I->getOpcode()) {
  case Instruction::Store: {
    auto S = cast<StoreInst>(I);
    if (isABIRegister(S->getPointerOperand())) {
      Result.push_back(ABIRegisters.lookup(
        cast<GlobalVariable>(S->getPointerOperand())));
    }
    break;
  }
  }
  return Result;
}

inline SmallVector<Register, 1>
ABIAnalysis::getRegistersRead(const Instruction *I) const {
  SmallVector<Register, 1> Result;
  switch (I->getOpcode()) {
  case Instruction::Load: {
    auto L = cast<LoadInst>(I);
    if (isABIRegister(L->getPointerOperand())) {
      Result.push_back(ABIRegisters.lookup(
        cast<GlobalVariable>(L->getPointerOperand())));
    }
    break;
  }
  }
  return Result;
}
} // namespace ABIAnalyses
