#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/IR/Module.h"
#include "llvm/Pass.h"

/// Inline every `revng_inline` helper at its call site in the `Isolated`
/// functions, where the critical arguments are constant at the call site. Does
/// not link helper bodies (use `LinkHelpersToInlinePass` first) and does not
/// delete inlined helper bodies (use `DeleteHelperBodiesPass` once at the end
/// of the pipeline).
class InlineHelpersPass : public llvm::ModulePass {
public:
  static char ID;

public:
  InlineHelpersPass() : ModulePass(ID) {}

  bool runOnModule(llvm::Module &M) override;
};
