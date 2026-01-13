#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/Pass.h"

#include "revng/BasicAnalyses/GeneratedCodeBasicInfo.h"

class InlineHelpersPass : public llvm::ModulePass {
public:
  static char ID;

  // This map stores the loaded helper module(s) to be inlined. Generally state
  // in a pass is frowned upon, but since this is only ever read it's simpler to
  // store it at the pass level.
  llvm::StringMap<std::unique_ptr<llvm::Module>> HelpersModules;

public:
  InlineHelpersPass() : ModulePass(ID) {}

  bool runOnModule(llvm::Module &M) override;

private:
  void linkRequiredHelpers(llvm::Module &M);
};
