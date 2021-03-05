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

/// Run all abi analyses on the oulined function F
/// the outlined function must have all original function calls
/// replaced with a basic block starting with a call to @precall_hook
/// followed by a summary of the side effects of the function
/// followed by a call to @postcall_hook and a basic block terminating
/// instruction
inline void
analyzeOutlinedFunction(Function *F, const GeneratedCodeBasicInfo &GCBI) {
  // find summary blocks
  F->print(llvm::errs());
  llvm::errs() << '\n';

  llvm::errs() << "------- start UsedArgumentsOfFunction --------\n";
  for (auto &[GV, State] :
       UsedArgumentsOfFunction::analyze(&F->getEntryBlock(), GCBI)) {
    llvm::errs() << GV->getName() << " = "
                 << model::RegisterState::getName(State) << "\n";
  }
  llvm::errs() << "------- end UsedArgumentsOfFunction --------\n";
  llvm::errs() << "------- start DeadRegisterArgumentsOfFunction --------\n";
  for (auto &[GV, State] :
       DeadRegisterArgumentsOfFunction::analyze(&F->getEntryBlock(), GCBI)) {
    llvm::errs() << GV->getName() << " = "
                 << model::RegisterState::getName(State) << "\n";
  }
  llvm::errs() << "------- end DeadRegisterArgumentsOfFunction --------\n";
  for (auto &BB : *F) {
    for (auto &I : BB) {
      if (auto *C = llvm::dyn_cast<llvm::CallInst>(&I)) {
        if (C->getCalledFunction()->getName() == "precall_hook") {
          llvm::errs() << *C << '\n';
          llvm::errs() << "------- start UsedReturnValuesOfFunctionCall "
                          "--------\n";
          for (auto &[GV, State] :
               UsedReturnValuesOfFunctionCall::analyze(&BB, GCBI)) {
            llvm::errs() << GV->getName() << " = "
                         << model::RegisterState::getName(State) << "\n";
          }
          llvm::errs() << "------- end UsedReturnValuesOfFunctionCall "
                          "--------\n";
          llvm::errs() << "------- start RegisterArgumentsOfFunctionCall "
                          "--------\n";
          for (auto &[GV, State] :
               RegisterArgumentsOfFunctionCall::analyze(&BB, GCBI)) {
            llvm::errs() << GV->getName() << " = "
                         << model::RegisterState::getName(State) << "\n";
          }
          llvm::errs() << "------- end RegisterArgumentsOfFunctionCall "
                          "--------\n";
          llvm::errs() << "------- start DeadReturnValuesOfFunctionCall "
                          "--------\n";
          for (auto &[GV, State] :
               DeadReturnValuesOfFunctionCall::analyze(&BB, GCBI)) {
            llvm::errs() << GV->getName() << " = "
                         << model::RegisterState::getName(State) << "\n";
          }
          llvm::errs() << "------- end DeadReturnValuesOfFunctionCall "
                          "--------\n";
        } else if (C->getCalledFunction()->getName() == "postcall_hook") {
          llvm::errs() << *C << '\n';
        }
      } else if (auto *R = llvm::dyn_cast<llvm::ReturnInst>(&I)) {
        llvm::errs() << "------- start UsedReturnValuesOfFunction --------\n";
        for (auto &[GV, State] :
             UsedReturnValuesOfFunction::analyze(&BB, GCBI)) {
          llvm::errs() << GV->getName() << " = "
                       << model::RegisterState::getName(State) << "\n";
        }
        llvm::errs() << "------- end UsedReturnValuesOfFunction --------\n";
        llvm::errs() << *R << '\n';
      }
    }
    // BB is definitely a call site and also a special basic block
  }
}

} // namespace ABIAnalyses
