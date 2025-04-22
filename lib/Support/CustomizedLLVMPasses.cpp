//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/Passes/PassBuilder.h"
#include "llvm/Transforms/InstCombine/InstCombine.h"
#include "llvm/Transforms/InstCombine/InstCombiner.h"
#include "llvm/Transforms/Scalar/SROA.h"
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

//
// Customized version of SROA, to disable flattening of arrays
//

class SROANoArraysPass : public FunctionPass {
public:
  static char ID;

  SROANoArraysPass() : FunctionPass(ID) {}

  void getAnalysisUsage(AnalysisUsage &AU) const override {}

  bool runOnFunction(Function &F) override {
    // Temporarily swap out the SROANoArray cl::opt and force it to true, so
    // that when the SROAPass runs and reads it it doesn't replace loads and
    // stores to array-typed allocas.
    // This is a pattern we need to prevent SROA to optimize away, because it
    // replaces multi-byte memory accesses with many single-byte accesses,
    // which is detrimental for information we need to recover about
    // memory accesses in the analyzed binary program.
    bool OriginalSROANoArrays = SROANoArrays;
    SROANoArrays = true;

    FunctionPassManager FPM;
    FPM.addPass(SROAPass(SROAOptions::PreserveCFG));

    FunctionAnalysisManager FAM;

    PassBuilder PB;
    PB.registerFunctionAnalyses(FAM);

    FPM.run(F, FAM);

    SROANoArrays = OriginalSROANoArrays;
    return true;
  }
};

char SROANoArraysPass::ID = 0;

static RegisterPass<SROANoArraysPass> RSROA("sroa-noarrays", "", false, false);

//
// Customized InstructionCombiningPass, to disable flattening of arrays
//

class InstCombineNoArrays : public InstructionCombiningPass {
public:
  static char ID;

  InstCombineNoArrays() : InstructionCombiningPass() {}

  bool runOnFunction(Function &F) override {
    // Temporarily swap out the MaxArraySize cl::opt and force it to 0, so that
    // when the InstCombinePass runs and reads it it doesn't replace loads and
    // stores to array-typed allocas.
    // This is a pattern we need to prevent instcombine to optimize away,
    // because it replaces multi-byte memory accesses with many single-byte
    // accesses, which is detrimental for information we need to recover about
    // memory accesses in the analyzed binary program.
    unsigned OriginalMaxArraySize = MaxArraySize;
    MaxArraySize = 0;

    InstructionCombiningPass::runOnFunction(F);

    MaxArraySize = OriginalMaxArraySize;
    return true;
  }
};

char InstCombineNoArrays::ID;

using RegisterInstCombine = RegisterPass<InstCombineNoArrays>;
static RegisterInstCombine RIC("instcombine-noarrays", "", false, false);
