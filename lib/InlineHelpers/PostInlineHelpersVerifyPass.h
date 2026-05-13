#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/Pass.h"

/// Module pass that asserts no `getelementptr` instruction remains in any
/// `Isolated` function after the `inline-helpers` pass has run.
class PostInlineHelpersVerifyPass : public llvm::ModulePass {
public:
  static char ID;

public:
  PostInlineHelpersVerifyPass() : llvm::ModulePass(ID) {}

public:
  bool runOnModule(llvm::Module &M) final;
};
