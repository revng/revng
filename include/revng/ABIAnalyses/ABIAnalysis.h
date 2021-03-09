#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/IR/Function.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/raw_ostream.h"

#include "revng/ABIAnalyses/Analyses.h"
#include "revng/Model/Binary.h"
#include "revng/Support/Debug.h"

namespace ABIAnalyses {
using llvm::CallInst;
using llvm::dyn_cast;
using llvm::Function;
using llvm::ReturnInst;

inline Logger<> ABILogger("new-abi");

/// Run all abi analyses on the oulined function F
/// the outlined function must have all original function calls
/// replaced with a basic block starting with a call to @precall_hook
/// followed by a summary of the side effects of the function
/// followed by a call to @postcall_hook and a basic block terminating
/// instruction
inline void
analyzeOutlinedFunction(Function *F, const GeneratedCodeBasicInfo &GCBI) {
  // find summary blocks
  revng_log(ABILogger, "Analyzing function:\n" << F);
  ABILogger << "------- start UsedArgumentsOfFunction --------\n";
  for (auto &[GV, State] :
       UsedArgumentsOfFunction::analyze(&F->getEntryBlock(), GCBI)) {
    ABILogger << GV->getName() << " = " << model::RegisterState::getName(State)
              << "\n";
  }
  ABILogger << "------- end UsedArgumentsOfFunction --------\n";
  ABILogger << "------- start DeadRegisterArgumentsOfFunction --------\n";
  for (auto &[GV, State] :
       DeadRegisterArgumentsOfFunction::analyze(&F->getEntryBlock(), GCBI)) {
    ABILogger << GV->getName() << " = " << model::RegisterState::getName(State)
              << "\n";
  }
  ABILogger << "------- end DeadRegisterArgumentsOfFunction --------\n";

  for (auto &BB : *F) {
    for (auto &I : BB) {
      if (auto *C = dyn_cast<CallInst>(&I)) {
        if (C->getCalledFunction()->getName().contains("precall_hook")) {
          ABILogger << C << '\n';
          ABILogger << "------- start UsedReturnValuesOfFunctionCall "
                       "--------\n";
          for (auto &[GV, State] :
               UsedReturnValuesOfFunctionCall::analyze(&BB, GCBI)) {
            ABILogger << GV->getName() << " = "
                      << model::RegisterState::getName(State) << "\n";
          }
          ABILogger << "------- end UsedReturnValuesOfFunctionCall "
                       "--------\n";

          ABILogger << "------- start RegisterArgumentsOfFunctionCall "
                       "--------\n";
          for (auto &[GV, State] :
               RegisterArgumentsOfFunctionCall::analyze(&BB, GCBI)) {
            ABILogger << GV->getName() << " = "
                      << model::RegisterState::getName(State) << "\n";
          }
          ABILogger << "------- end RegisterArgumentsOfFunctionCall "
                       "--------\n";
          ABILogger << "------- start DeadReturnValuesOfFunctionCall "
                       "--------\n";
          for (auto &[GV, State] :
               DeadReturnValuesOfFunctionCall::analyze(&BB, GCBI)) {
            ABILogger << GV->getName() << " = "
                      << model::RegisterState::getName(State) << "\n";
          }
          ABILogger << "------- end DeadReturnValuesOfFunctionCall "
                       "--------\n";
        }
      } else if (auto *R = dyn_cast<ReturnInst>(&I)) {
        ABILogger << "------- start UsedReturnValuesOfFunction --------\n";
        for (auto &[GV, State] :
             UsedReturnValuesOfFunction::analyze(&BB, GCBI)) {
          ABILogger << GV->getName() << " = "
                    << model::RegisterState::getName(State) << "\n";
        }
        ABILogger << "------- end UsedReturnValuesOfFunction --------\n";
      }
    }
    // BB is definitely a call site and also a special basic block
  }
}

} // namespace ABIAnalyses
