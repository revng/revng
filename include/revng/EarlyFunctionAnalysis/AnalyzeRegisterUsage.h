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
#include "revng/Support/MetaAddress.h"

namespace efa {

// TODO: switch to model::Register?
using CSVSet = std::set<llvm::GlobalVariable *>;

class RUAResults {
public:
  struct CallSiteResults {
    MetaAddress CalleeAddress;
    CSVSet ArgumentsRegisters;
    CSVSet ReturnValuesRegisters;

    bool operator==(const CallSiteResults &Other) const = default;
  };

public:
  // Per function analysis
  CSVSet ArgumentsRegisters;
  CSVSet ReturnValuesRegisters;

  // Per call site analysis
  std::map<BasicBlockID, CallSiteResults> CallSites;

public:
  bool operator==(const RUAResults &Other) const = default;

public:
  void combine(const RUAResults &Other) {
    auto HandleMap = [](CSVSet &ThisSet, const CSVSet &OtherSet) {
      for (auto *CSV : OtherSet)
        ThisSet.insert(CSV);
    };

    HandleMap(ArgumentsRegisters, Other.ArgumentsRegisters);
    HandleMap(ReturnValuesRegisters, Other.ReturnValuesRegisters);

    for (auto &[OtherBlockID, OtherResults] : Other.CallSites) {
      HandleMap(CallSites[OtherBlockID].ArgumentsRegisters,
                OtherResults.ArgumentsRegisters);
      HandleMap(CallSites[OtherBlockID].ReturnValuesRegisters,
                OtherResults.ReturnValuesRegisters);
    }
  }

public:
  // Debug methods
  void dump() const debug_function { dump(dbg, ""); }

  template<typename T>
  void dump(T &Output, const char *Prefix = "") const;
};

extern template void RUAResults::dump<Logger>(Logger &, const char *) const;

RUAResults analyzeRegisterUsage(llvm::Function *F,
                                const GeneratedCodeBasicInfo &,
                                model::Architecture::Values Architecture,
                                llvm::Function *,
                                llvm::Function *,
                                llvm::Function *);

} // namespace efa
