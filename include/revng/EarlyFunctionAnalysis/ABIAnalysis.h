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

#include "revng/ABI/RegisterState.h"
#include "revng/BasicAnalyses/GeneratedCodeBasicInfo.h"
#include "revng/Model/Binary.h"
#include "revng/Support/Debug.h"
#include "revng/Support/MetaAddress.h"

namespace ABIAnalyses {

using RegisterStateMap = std::map<const llvm::GlobalVariable *,
                                  abi::RegisterState::Values>;

struct ABIAnalysesResults {
  // Per function analysis
  RegisterStateMap ArgumentsRegisters;

  // Per call site analysis
  struct CallSiteResults {
    RegisterStateMap ArgumentsRegisters;
    RegisterStateMap ReturnValuesRegisters;
  };
  std::map<BasicBlockID, CallSiteResults> CallSites;

  // Per return analysis
  std::map<BasicBlockID, RegisterStateMap> ReturnValuesRegisters;
  RegisterStateMap FinalReturnValuesRegisters;

  // Debug methods
  void dump() const debug_function { dump(dbg, ""); }

  template<typename T>
  void dump(T &Output, const char *Prefix = "") const;
};

extern template void
ABIAnalyses::ABIAnalysesResults::dump<Logger<true>>(Logger<true> &,
                                                    const char *) const;

abi::RegisterState::Values combine(abi::RegisterState::Values,
                                   abi::RegisterState::Values);

ABIAnalysesResults analyzeOutlinedFunction(llvm::Function *F,
                                           const GeneratedCodeBasicInfo &,
                                           llvm::Function *,
                                           llvm::Function *,
                                           llvm::Function *);

void finalizeReturnValues(ABIAnalysesResults &);

} // namespace ABIAnalyses
