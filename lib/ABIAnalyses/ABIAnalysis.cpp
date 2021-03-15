//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/IR/Function.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/raw_ostream.h"

#include "revng/ABIAnalyses/ABIAnalysis.h"
#include "revng/ABIAnalyses/Common.h"
#include "revng/Model/Binary.h"
#include "revng/Support/Debug.h"

#include "Analyses.h"

namespace ABIAnalyses {
using namespace llvm;

/// Run all abi analyses on the oulined function F the outlined function must
/// have all original function calls replaced with a basic block starting with a
/// call to @precall_hook followed by a summary of the side effects of the
/// function followed by a call to @postcall_hook and a basic block terminating
/// instruction
AnalysisResults
analyzeOutlinedFunction(Function *F, const GeneratedCodeBasicInfo &GCBI) {
  AnalysisResults Results;

  Results.UsedArgumentsOfFunction = UsedArgumentsOfFunction::
    analyze(&F->getEntryBlock(), GCBI);
  Results.DeadRegisterArgumentsOfFunction = DeadRegisterArgumentsOfFunction::
    analyze(&F->getEntryBlock(), GCBI);
  for (Instruction &I : llvm::instructions(F)) {
    BasicBlock *BB = I.getParent();
    if (auto *C = dyn_cast<CallInst>(&I)) {
      if (C->getCalledFunction()->getName().contains("precall_hook")) {
        // BB is definitely a call site and also a special basic block
        Results.UsedReturnValuesOfFunctionCall
          [BB] = UsedReturnValuesOfFunctionCall::analyze(BB, GCBI);
        Results.RegisterArgumentsOfFunctionCall
          [BB] = RegisterArgumentsOfFunctionCall::analyze(BB, GCBI);
        Results.DeadReturnValuesOfFunctionCall
          [BB] = DeadReturnValuesOfFunctionCall::analyze(BB, GCBI);
      }
    } else if (auto *R = dyn_cast<ReturnInst>(&I)) {
      Results.UsedReturnValuesOfFunction[BB] = UsedReturnValuesOfFunction::
        analyze(BB, GCBI);
    }
  }

  return Results;
}

} // namespace ABIAnalyses
