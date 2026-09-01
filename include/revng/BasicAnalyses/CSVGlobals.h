#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/IR/Module.h"

#include "revng/Model/Binary.h"

/// Information about the CPU state variables in a generated module.
class CSVGlobals {
private:
  llvm::GlobalVariable *PC = nullptr;
  llvm::GlobalVariable *SP = nullptr;
  llvm::GlobalVariable *RA = nullptr;
  std::vector<llvm::GlobalVariable *> CSVs;
  std::vector<llvm::GlobalVariable *> ABIRegisters;
  std::set<llvm::GlobalVariable *> ABIRegistersSet;

public:
  CSVGlobals(const model::Binary &Binary, llvm::Module &M);

  llvm::GlobalVariable *pcReg() const { return PC; }
  llvm::GlobalVariable *spReg() const { return SP; }
  llvm::GlobalVariable *raReg() const { return RA; }

  bool isPCReg(const llvm::GlobalVariable *GV) const;
  bool isSPReg(const llvm::GlobalVariable *GV) const;
  bool isSPReg(const llvm::Value *V) const;

  llvm::ArrayRef<llvm::GlobalVariable *> csvs() const { return CSVs; }

  const std::vector<llvm::GlobalVariable *> &abiRegisters() const {
    return ABIRegisters;
  }

  bool isABIRegister(llvm::GlobalVariable *CSV) const {
    return ABIRegistersSet.contains(CSV);
  }
};
