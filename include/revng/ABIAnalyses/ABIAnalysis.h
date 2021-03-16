#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/IR/Function.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/raw_ostream.h"

#include "revng/BasicAnalyses/GeneratedCodeBasicInfo.h"
#include "revng/Model/Binary.h"
#include "revng/Support/Debug.h"

namespace ABIAnalyses {

using RegisterStateMap = llvm::DenseMap<const llvm::GlobalVariable *,
                                        model::RegisterState::Values>;
struct AnalysisResults {
  // Per function analysis
  RegisterStateMap UAOF;
  RegisterStateMap DRAOF;

  // Per call site analysis
  std::map<llvm::BasicBlock *, RegisterStateMap> URVOFC;
  std::map<llvm::BasicBlock *, RegisterStateMap> RAOFC;
  std::map<llvm::BasicBlock *, RegisterStateMap> DRVOFC;

  // Per return analysis
  std::map<llvm::BasicBlock *, RegisterStateMap> URVOF;

  // Debug methods
  void dump() const debug_function { dump(dbg, ""); }

  template<typename T>
  void dump(T &Output, const char *Prefix) const;
};

/// Run all abi analyses on the oulined function F the outlined function must
/// have all original function calls replaced with a basic block starting with a
/// call to @precall_hook followed by a summary of the side effects of the
/// function followed by a call to @postcall_hook and a basic block terminating
/// instruction
AnalysisResults
analyzeOutlinedFunction(llvm::Function *F, const GeneratedCodeBasicInfo &GCBI);

/// Print the analysis results
template<typename T>
void AnalysisResults::dump(T &Output, const char *Prefix) const {
  Output << Prefix << "UsedArgumentsOfFunction:\n";
  for (auto &[GV, State] : UAOF) {
    Output << Prefix << "  " << GV->getName().str() << " = "
           << model::RegisterState::getName(State).str() << '\n';
  }
  Output << Prefix << "DeadRegisterArgumentsOfFunction:\n";
  for (auto &[GV, State] : DRAOF) {
    Output << Prefix << "  " << GV->getName().str() << " = "
           << model::RegisterState::getName(State).str() << '\n';
  }

  Output << Prefix << "UsedReturnValuesOfFunctionCall:\n";
  for (auto &[BB, StateMap] : URVOFC) {
    Output << Prefix << "  " << BB->getName().str() << '\n';
    for (auto &[GV, State] : StateMap) {
      Output << Prefix << "    " << GV->getName().str() << " = "
             << model::RegisterState::getName(State).str() << '\n';
    }
  }

  Output << Prefix << "RegisterArgumentsOfFunctionCall:\n";
  for (auto &[BB, StateMap] : RAOFC) {
    Output << Prefix << "  " << BB->getName().str() << '\n';
    for (auto &[GV, State] : StateMap) {
      Output << Prefix << "    " << GV->getName().str() << " = "
             << model::RegisterState::getName(State).str() << '\n';
    }
  }

  Output << Prefix << "DeadReturnValuesOfFunctionCall:\n";
  for (auto &[BB, StateMap] : DRVOFC) {
    Output << Prefix << "  " << BB->getName().str() << '\n';
    for (auto &[GV, State] : StateMap) {
      Output << Prefix << "    " << GV->getName().str() << " = "
             << model::RegisterState::getName(State).str() << '\n';
    }
  }

  Output << Prefix << "UsedReturnValuesOfFunction:\n";
  for (auto &[BB, StateMap] : URVOF) {
    Output << Prefix << "  " << BB->getName().str() << '\n';
    for (auto &[GV, State] : StateMap) {
      Output << Prefix << "    " << GV->getName().str() << " = "
             << model::RegisterState::getName(State).str() << '\n';
    }
  }
}

} // namespace ABIAnalyses
