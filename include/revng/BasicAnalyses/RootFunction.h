#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/Function.h"

#include "revng/Support/MetaAddress.h"

/// Lazily collected information about the generated root function.
class RootFunction {
private:
  llvm::Function *TheFunction = nullptr;
  llvm::Function *NewPC = nullptr;
  llvm::BasicBlock *Dispatcher = nullptr;
  llvm::BasicBlock *DispatcherFail = nullptr;
  llvm::BasicBlock *AnyPC = nullptr;
  llvm::BasicBlock *UnexpectedPC = nullptr;
  std::map<MetaAddress, llvm::BasicBlock *> JumpTargets;

public:
  llvm::Function *getFunction() { return TheFunction; }
  llvm::BasicBlock *anyPC() { return AnyPC; }
  llvm::BasicBlock *unexpectedPC() { return UnexpectedPC; }
  llvm::BasicBlock *dispatcher() { return Dispatcher; }

public:
  explicit RootFunction(llvm::Module &M);

  /// Return the basic block associated to \p PC.
  ///
  /// Returns nullptr if the PC doesn't have a basic block.
  llvm::BasicBlock *getBlockAt(MetaAddress PC);

  bool isJump(llvm::BasicBlock *BB);

  /// Return true if \p T represents a jump in the input assembly.
  ///
  /// Return true if \p T targets include only dispatcher-related basic blocks
  /// and jump targets.
  bool isJump(llvm::Instruction *T);
};
