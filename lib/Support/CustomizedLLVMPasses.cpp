//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/Passes/PassBuilder.h"
#include "llvm/Transforms/Scalar/SimplifyCFG.h"

using namespace llvm;

//
// Customized version of SimplifyCFG, with a different set of default options.
//

class SimplifyCFGWithHoistAndSinkPass : public FunctionPass {
public:
  static char ID;
  SimplifyCFGWithHoistAndSinkPass() : FunctionPass(ID) {}

  void getAnalysisUsage(AnalysisUsage &AU) const override {}

  bool runOnFunction(Function &F) override {
    FunctionPassManager FPM;
    FPM.addPass(SimplifyCFGPass(SimplifyCFGOptions()
                                  .convertSwitchRangeToICmp(true)
                                  .hoistCommonInsts(true)
                                  .sinkCommonInsts(true)));

    FunctionAnalysisManager FAM;

    PassBuilder PB;
    PB.registerFunctionAnalyses(FAM);

    FPM.run(F, FAM);
    return true;
  }
};

char SimplifyCFGWithHoistAndSinkPass::ID;

using RegisterCustomSimplifyCFG = RegisterPass<SimplifyCFGWithHoistAndSinkPass>;
static RegisterCustomSimplifyCFG
  RSCG("simplify-cfg-with-hoist-and-sink", "", false, false);
