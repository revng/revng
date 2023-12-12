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

class ABIAnalysesResults {
public:
  struct CallSiteResults {
    MetaAddress CalleeAddress;
    RegisterStateMap ArgumentsRegisters;
    RegisterStateMap ReturnValuesRegisters;

    bool operator==(const CallSiteResults &Other) const = default;
  };

public:
  // Per function analysis
  RegisterStateMap ArgumentsRegisters;

  // Per call site analysis
  std::map<BasicBlockID, CallSiteResults> CallSites;

  // Per return analysis
  std::map<BasicBlockID, RegisterStateMap> ReturnValuesRegisters;
  RegisterStateMap FinalReturnValuesRegisters;

public:
  bool operator==(const ABIAnalysesResults &Other) const = default;

public:
  void combine(const ABIAnalysesResults &Other) {
    auto HandleMap = [](RegisterStateMap &ThisMap,
                        const RegisterStateMap &OtherMap) {
      for (auto &[GV, Value] : OtherMap) {
        if (Value == abi::RegisterState::Yes)
          ThisMap[GV] = Value;
      }
    };

    HandleMap(ArgumentsRegisters, Other.ArgumentsRegisters);
    HandleMap(FinalReturnValuesRegisters, Other.FinalReturnValuesRegisters);

    for (auto &[OtherBlockID, OtherMap] : Other.ReturnValuesRegisters)
      HandleMap(ReturnValuesRegisters[OtherBlockID], OtherMap);

    for (auto &[OtherBlockID, OtherResults] : Other.CallSites) {
      HandleMap(CallSites[OtherBlockID].ArgumentsRegisters,
                OtherResults.ArgumentsRegisters);
      HandleMap(CallSites[OtherBlockID].ReturnValuesRegisters,
                OtherResults.ReturnValuesRegisters);
    }
  }

  void normalize() {
    auto HandleMap = [](RegisterStateMap &Map) {
      for (auto &[CSV, Value] : Map) {
        switch (Value) {
        case abi::RegisterState::Yes:
        case abi::RegisterState::YesOrDead:
        case abi::RegisterState::Dead:
          Value = abi::RegisterState::Yes;
          break;
        case abi::RegisterState::No:
        case abi::RegisterState::NoOrDead:
        case abi::RegisterState::Maybe:
        case abi::RegisterState::Contradiction:
          Value = abi::RegisterState::No;
          break;
        case abi::RegisterState::Invalid:
        case abi::RegisterState::Count:
          revng_abort();
        }
      }
    };

    HandleMap(ArgumentsRegisters);
    HandleMap(FinalReturnValuesRegisters);

    for (auto &[BlockID, Map] : ReturnValuesRegisters)
      HandleMap(Map);

    for (auto &[BlockID, Results] : CallSites) {
      HandleMap(Results.ArgumentsRegisters);
      HandleMap(Results.ReturnValuesRegisters);
    }
  }

public:
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
