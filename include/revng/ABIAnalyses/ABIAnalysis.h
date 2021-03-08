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

namespace ABIAnalyses {
using llvm::Function;
using llvm::errs;
using llvm::dyn_cast;
using llvm::CallInst;
using llvm::ReturnInst;
/// Run all abi analyses on the oulined function F
/// the outlined function must have all original function calls
/// replaced with a basic block starting with a call to @precall_hook
/// followed by a summary of the side effects of the function
/// followed by a call to @postcall_hook and a basic block terminating
/// instruction
inline void
analyzeOutlinedFunction(Function *F, const GeneratedCodeBasicInfo &GCBI) {
  // find summary blocks
  F->print(errs());
  errs() << '\n';

  errs() << "------- start UsedArgumentsOfFunction --------\n";
  for (auto &[GV, State] :
       UsedArgumentsOfFunction::analyze(&F->getEntryBlock(), GCBI)) {
    errs() << GV->getName() << " = "
                 << model::RegisterState::getName(State) << "\n";
  }
  errs() << "------- end UsedArgumentsOfFunction --------\n";
  errs() << "------- start DeadRegisterArgumentsOfFunction --------\n";
  for (auto &[GV, State] :
       DeadRegisterArgumentsOfFunction::analyze(&F->getEntryBlock(), GCBI)) {
    errs() << GV->getName() << " = "
                 << model::RegisterState::getName(State) << "\n";
  }
  errs() << "------- end DeadRegisterArgumentsOfFunction --------\n";
  for (auto &BB : *F) {
    for (auto &I : BB) {
      if (auto *C = dyn_cast<CallInst>(&I)) {
        if (C->getCalledFunction()->getName() == "precall_hook") {
          errs() << *C << '\n';
          errs() << "------- start UsedReturnValuesOfFunctionCall "
                          "--------\n";
          for (auto &[GV, State] :
               UsedReturnValuesOfFunctionCall::analyze(&BB, GCBI)) {
            errs() << GV->getName() << " = "
                         << model::RegisterState::getName(State) << "\n";
          }
          errs() << "------- end UsedReturnValuesOfFunctionCall "
                          "--------\n";
          errs() << "------- start RegisterArgumentsOfFunctionCall "
                          "--------\n";
          for (auto &[GV, State] :
               RegisterArgumentsOfFunctionCall::analyze(&BB, GCBI)) {
            errs() << GV->getName() << " = "
                         << model::RegisterState::getName(State) << "\n";
          }
          errs() << "------- end RegisterArgumentsOfFunctionCall "
                          "--------\n";
          errs() << "------- start DeadReturnValuesOfFunctionCall "
                          "--------\n";
          for (auto &[GV, State] :
               DeadReturnValuesOfFunctionCall::analyze(&BB, GCBI)) {
            errs() << GV->getName() << " = "
                         << model::RegisterState::getName(State) << "\n";
          }
          errs() << "------- end DeadReturnValuesOfFunctionCall "
                          "--------\n";
        } else if (C->getCalledFunction()->getName() == "postcall_hook") {
          errs() << *C << '\n';
        }
      } else if (auto *R = dyn_cast<ReturnInst>(&I)) {
        errs() << "------- start UsedReturnValuesOfFunction --------\n";
        for (auto &[GV, State] :
             UsedReturnValuesOfFunction::analyze(&BB, GCBI)) {
          errs() << GV->getName() << " = "
                       << model::RegisterState::getName(State) << "\n";
        }
        errs() << "------- end UsedReturnValuesOfFunction --------\n";
        errs() << *R << '\n';
      }
    }
    // BB is definitely a call site and also a special basic block
  }
}

} // namespace ABIAnalyses
