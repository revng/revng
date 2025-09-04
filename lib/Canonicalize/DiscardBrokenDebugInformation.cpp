//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/IR/Function.h"
#include "llvm/Pass.h"

#include "revng/Pipes/IRHelpers.h"
#include "revng/Support/Debug.h"
#include "revng/Support/Error.h"
#include "revng/Support/IRHelpers.h"

static Logger<> Log("discarded-debug-information");

struct DiscardBrokenDebugInformation : public llvm::FunctionPass {
public:
  static char ID;

  DiscardBrokenDebugInformation() : llvm::FunctionPass(ID) {}

  bool runOnFunction(llvm::Function &F) override {
    bool WasModified = false;
    for (llvm::BasicBlock &BB : F) {
      for (llvm::Instruction &I : BB) {
        if (llvm::Error Error = isDebugLocationInvalid(I)) {
          std::string ErrorMessage = revng::unwrapError(std::move(Error));
          revng_log(Log,
                    "Discarding debug information from:\n"
                      << dumpToString(I) << '\n');
          revng_log(Log, ErrorMessage << '\n');

          I.setDebugLoc({});
          WasModified = true;
        }
      }
    }

    return WasModified;
  }

  void getAnalysisUsage(llvm::AnalysisUsage &AU) const override {
    AU.setPreservesCFG();
  }
};

char DiscardBrokenDebugInformation::ID = 0;

using RBDI = DiscardBrokenDebugInformation;
llvm::RegisterPass<RBDI> R("discard-broken-debug-information",
                           "Sometimes llvm passes break the debug information "
                           "we use to maintain the link between the decompiled "
                           "code and the original assembly. This pass detects "
                           "such broken values and removes them.",
                           false,
                           false);
