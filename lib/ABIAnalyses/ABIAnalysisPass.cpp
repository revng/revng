//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/IR/Function.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/Support/CommandLine.h"

#include "revng/ABIAnalyses/ABIAnalysisPass.h"
#include "revng/ABIAnalyses/Generated/DeadRegisterArgumentsOfFunction.h"
#include "revng/ABIAnalyses/Generated/DeadReturnValuesOfFunctionCall.h"
#include "revng/ABIAnalyses/Generated/RegisterArgumentsOfFunctionCall.h"
#include "revng/ABIAnalyses/Generated/UsedArgumentsOfFunction.h"
#include "revng/ABIAnalyses/Generated/UsedReturnValuesOfFunctionCall.h"

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

  errs() << F.getName().str() << '\n';
  errs() << "---------------- START DeadReturnValuesOfFunctionCall "
            "----------------\n";
  for (auto &B : F) {
    for (auto &I : B) {
      if (I.getOpcode() == Instruction::Call) {

        auto Result = DeadReturnValuesOfFunctionCall::analyze(&I, &F.getEntryBlock(), GCBI);
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

        auto Result = UsedReturnValuesOfFunctionCall::analyze(&I, &F.getEntryBlock(), GCBI);
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

  {
    errs() << "---------------- START DeadRegisterArgumentsOfFunction "
              "----------------\n";
    auto Result = DeadRegisterArgumentsOfFunction::analyze(nullptr, &F.getEntryBlock(), GCBI);
    errs() << "---------------- RESULTS ----------------\n";

    for (auto &Reg : Result) {
      errs() << "NoOrDead " << Reg.first->getName().str() << '\n';
    }
    errs() << "---------------- END DeadRegisterArgumentsOfFunction "
              "-------------------\n";
  }

  {
    errs() << "---------------- START UsedArgumentsOfFunction "
              "----------------\n";

    auto Result = UsedArgumentsOfFunction::analyze(nullptr, &F.getEntryBlock(), GCBI);
    errs() << "---------------- RESULTS ----------------\n";

    for (auto &Reg : Result) {
      errs() << "Yes " << Reg.first->getName().str() << '\n';
    }
    errs() << "---------------- END UsedArgumentsOfFunction "
              "-------------------\n";
  }

  errs() << "---------------- START RegisterArgumentsOfFunctionCall "
            "----------------\n";
  for (auto &B : F) {
    for (auto &I : B) {
      if (I.getOpcode() == Instruction::Call) {

        auto Result = RegisterArgumentsOfFunctionCall::analyze(&I,
                                                               &F.getEntryBlock(),
                                                               GCBI);
        errs() << "---------------- RESULTS ";
        I.print(errs());
        errs() << " ----------------\n";

        for (auto &Reg : Result) {
          errs() << "Yes " << Reg.first->getName().str() << '\n';
        }
      }
    }
  }
  errs() << "---------------- END RegisterArgumentsOfFunctionCall "
            "-------------------\n";

  return false;
}
