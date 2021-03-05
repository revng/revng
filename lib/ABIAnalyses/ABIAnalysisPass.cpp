//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/IR/Function.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/Support/CommandLine.h"

#include "revng/ABIAnalyses/ABIAnalysisPass.h"
#include "revng/ABIAnalyses/ABIAnalysis.h"

using namespace llvm;

static cl::OptionCategory Category("new-abi-analysis");

static cl::opt<std::string> FilterFunction("new-abi-analysis-function",
                                           cl::desc("only show results for "
                                                    "this function"),
                                           cl::value_desc("new-abi-analysis-"
                                                          "function"),
                                           cl::cat(Category));
char ABIAnalysisPass::ID = 0;

static RegisterPass<ABIAnalysisPass>
  X("new-abi-analysis", "abianalysis Functions Pass", true, true);

bool ABIAnalysisPass::runOnFunction(Function &F) {
  // Retrieve analysis of the GeneratedCodeBasicInfo pass
  auto &GCBI = getAnalysis<GeneratedCodeBasicInfoWrapperPass>().getGCBI();
  if (F.getName().str().compare(FilterFunction) != 0) {
    return false;
  }
  ABIAnalyses::analyzeOutlinedFunction(&F, GCBI);
  return false;
}
