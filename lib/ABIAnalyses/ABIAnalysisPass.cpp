//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/IR/Function.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/Support/CommandLine.h"

#include "revng/ABIAnalyses/ABIAnalysisPass.h"
#include "revng/ABIAnalyses/DeadReturnValuesOfFunctionCall.h"

using namespace llvm;

static cl::OptionCategory RevNgCategory("revng new-abi-analysis");

static cl::opt<std::string> FilterFunction("new-abi-analysis-function",
                                           cl::desc("only show results for "
                                                    "this function"),
                                           cl::value_desc("new-abi-analysis-"
                                                          "function"),
                                           cl::cat(RevNgCategory));
char ABIAnalysisPass::ID = 0;

static llvm::RegisterPass<ABIAnalysisPass>
  X("new-abi-analysis", "abianalysis Functions Pass", true, true);

bool ABIAnalysisPass::runOnFunction(llvm::Function &F) {
  llvm::errs() << F.getName().str() << '\n';
  // Retrieve analysis of the GeneratedCodeBasicInfo pass
  auto &GCBI = getAnalysis<GeneratedCodeBasicInfoWrapperPass>().getGCBI();
  if (F.getName().str().compare(FilterFunction) != 0) {
    return false;
  }

  for (auto &B : F) {
    for (auto &I : B) {
      if (I.getOpcode() == llvm::Instruction::Call) {
        llvm::errs() << "---------------- START ----------------\n";

        auto Result = DeadReturnValuesOfFunctionCall::analyze(&I, &F, GCBI, {});
        llvm::errs() << "---------------- RESULTS ----------------\n";

        for (auto &Reg : Result) {
          llvm::errs() << "NoOrDead " << Reg.first->getName().str() << '\n';
        }
        llvm::errs() << "---------------- END -------------------\n";
      }
    }
  }
  return false;
}
