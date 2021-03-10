//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/IR/Function.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Support/CommandLine.h"

#include "revng/ABIAnalyses/ABIAnalysis.h"
#include "revng/ABIAnalyses/ABIAnalysisPass.h"
#include "revng/BasicAnalyses/GeneratedCodeBasicInfo.h"

using namespace ABIAnalyses;
using namespace llvm;

static cl::OptionCategory Category("new-abi-analysis");

static cl::opt<std::string> FilterFunction("new-abi-analysis-function",
                                           cl::desc("only show results for "
                                                    "this function"),
                                           cl::value_desc("new-abi-analysis-"
                                                          "function"),
                                           cl::cat(Category));
char ABIAnalysisWrapperPass::ID = 0;

static RegisterPass<ABIAnalysisWrapperPass>
  X("new-abi-analysis", "abianalysis Functions Pass", true, true);

bool ABIAnalysisWrapperPass::runOnFunction(Function &F) {
  // Retrieve analysis of the GeneratedCodeBasicInfo pass
  auto &GCBI = getAnalysis<GeneratedCodeBasicInfoWrapperPass>().getGCBI();
  if (F.getName() != FilterFunction) {
    return false;
  }
  ABIAnalyses::analyzeOutlinedFunction(&F, GCBI);
  return false;
}


PreservedAnalyses
ABIAnalysisPass::run(Function &F, FunctionAnalysisManager &FAM) {
  const auto &GCBI = FAM.getResult<GeneratedCodeBasicInfoAnalysis>(F);
  if (F.getName() != FilterFunction) {
    return PreservedAnalyses::all();
  }
  ABIAnalyses::analyzeOutlinedFunction(&F, GCBI);
  return PreservedAnalyses::all();
}
