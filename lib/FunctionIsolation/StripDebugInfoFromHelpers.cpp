//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/IR/DebugInfo.h"
#include "llvm/Pass.h"

#include "revng/Model/FunctionTags.h"
#include "revng/Pipeline/RegisterLLVMPass.h"

using namespace llvm;

/// Debug Info on helpers is not very useful after this phase, since the binary
/// cannot be run after this phase.
static bool stripDebugInfoFromHelpers(Module &M) {
  for (llvm::Function &F : M) {
    if (not FunctionTags::Isolated.isTagOf(&F)) {
      // TODO: If we really want to preserve this info, we should make proper
      // location here.
      stripDebugInfo(F);
    }
  }

  return true;
}

struct StripDebugInfoFromHelpersPass : public llvm::ModulePass {
public:
  static char ID;

  StripDebugInfoFromHelpersPass() : ModulePass(ID) {}

  bool runOnModule(Module &M) override { return stripDebugInfoFromHelpers(M); }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesAll();
  }
};

char StripDebugInfoFromHelpersPass::ID = 0;
static constexpr const char *Flag = "strip-debug-info-from-helpers";
using Register = RegisterPass<StripDebugInfoFromHelpersPass>;
static Register
  X(Flag, "Pass that removes debug info from helpers. ", false, false);
