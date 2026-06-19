#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/IR/Module.h"
#include "llvm/Pass.h"

/// Link the bodies of every `revng_inline`-tagged helper that's still a
/// declaration in `M`, cloning them from the per-architecture
/// `libtcg-helpers-to-inline-<arch>.bc` shipped with revng.
class LinkHelpersToInlinePass : public llvm::ModulePass {
public:
  static char ID;

public:
  LinkHelpersToInlinePass() : llvm::ModulePass(ID) {}

  bool runOnModule(llvm::Module &M) override;
};
