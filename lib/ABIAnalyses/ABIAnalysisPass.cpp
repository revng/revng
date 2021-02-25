//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/IR/Function.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/Support/CommandLine.h"

#include "revng/ABIAnalyses/ABIAnalysisPass.h"
#include "revng/ABIAnalyses/DeadRegisterArgumentsOfFunction.h"
#include "revng/ABIAnalyses/DeadReturnValuesOfFunctionCall.h"
#include "revng/ABIAnalyses/UsedReturnValuesOfFunctionCall.h"

using namespace llvm;

static cl::OptionCategory RevNgCategory("revng new-abi-analysis");

static cl::opt<std::string> FilterFunction("new-abi-analysis-function",
                                           cl::desc("only show results for "
                                                    "this function"),
                                           cl::value_desc("new-abi-analysis-"
                                                          "function"),
                                           cl::cat(RevNgCategory));
char ABIAnalysisPass::ID = 0;

static RegisterPass<ABIAnalysisPass>
  X("new-abi-analysis", "abianalysis Functions Pass", true, true);

bool ABIAnalysisPass::runOnFunction(Function &F) {
  // Retrieve analysis of the GeneratedCodeBasicInfo pass
  auto &GCBI = getAnalysis<GeneratedCodeBasicInfoWrapperPass>().getGCBI();
  if (F.getName().str().compare(FilterFunction) != 0) {
    return false;
  }

  errs() << F.getName().str() << '\n';
  errs() << "---------------- START DeadReturnValuesOfFunctionCall "
            "----------------\n";
  for (auto &B : F) {
    for (auto &I : B) {
      if (I.getOpcode() == Instruction::Call) {

        auto Result = DeadReturnValuesOfFunctionCall::analyze(&I, &F, GCBI, {});
        errs() << "---------------- RESULTS ";
        I.print(errs());
        errs() << " ----------------\n";

        for (auto &Reg : Result) {
          errs() << "NoOrDead " << Reg.first->getName().str() << '\n';
        }
      }
    }
  }
  errs() << "---------------- END DeadReturnValuesOfFunctionCall "
            "-------------------\n";

  errs() << "---------------- START UsedReturnValuesOfFunctionCall "
            "----------------\n";
  for (auto &B : F) {
    for (auto &I : B) {
      if (I.getOpcode() == Instruction::Call) {

        auto Result = UsedReturnValuesOfFunctionCall::analyze(&I, &F, GCBI, {});
        errs() << "---------------- RESULTS ";
        I.print(errs());
        errs() << " ----------------\n";

        for (auto &Reg : Result) {
          errs() << "Yes " << Reg.first->getName().str() << '\n';
        }
      }
    }
  }
  errs() << "---------------- END UsedReturnValuesOfFunctionCall "
            "-------------------\n";

  errs() << "---------------- START DeadRegisterArgumentsOfFunction "
            "----------------\n";

  auto Result = DeadRegisterArgumentsOfFunction::analyze(&F, GCBI, {});
  errs() << "---------------- RESULTS ----------------\n";

  for (auto &Reg : Result) {
    errs() << "NoOrDead " << Reg.first->getName().str() << '\n';
  }
  errs() << "---------------- END DeadRegisterArgumentsOfFunction "
            "-------------------\n";
  return false;
}
