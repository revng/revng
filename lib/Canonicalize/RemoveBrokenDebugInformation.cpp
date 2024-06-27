//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/IR/DebugInfoMetadata.h"
#include "llvm/IR/DebugLoc.h"
#include "llvm/IR/Function.h"
#include "llvm/Pass.h"

#include "revng/Pipeline/Location.h"
#include "revng/Pipes/Ranks.h"
#include "revng/Support/Debug.h"

struct RemoveBrokenDebugInformation : public llvm::FunctionPass {
public:
  static char ID;

  RemoveBrokenDebugInformation() : llvm::FunctionPass(ID) {}

  bool runOnFunction(llvm::Function &F) override {
    bool WasModified = false;
    for (llvm::BasicBlock &BB : F) {
      for (llvm::Instruction &I : BB) {
        if (I.getDebugLoc()) {
          if (I.getDebugLoc()->getInlinedAt() == nullptr) {
            I.setDebugLoc({});
            WasModified = true;

          } else if (VerifyLog.isEnabled()) {
            const auto &Serialized = I.getDebugLoc()->getScope()->getName();
            revng_assert(pipeline::locationFromString(revng::ranks::Instruction,
                                                      Serialized.str()));
          }
        }
      }
    }

    return WasModified;
  }

  void getAnalysisUsage(llvm::AnalysisUsage &AU) const override {
    AU.setPreservesCFG();
  }
};

char RemoveBrokenDebugInformation::ID = 0;

using RBDI = RemoveBrokenDebugInformation;
llvm::RegisterPass<RBDI> R("remove-broken-debug-information",
                           "Sometimes llvm passes break the debug information "
                           "we use to maintain the link between the decompiled "
                           "code and the original assembly. This pass detects "
                           "such broken values and removes them.",
                           false,
                           false);
