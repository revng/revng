#pragma once

/// \file InterproceduralAnalysis.h

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <set>
#include <variant>

#include "llvm/Support/DOTGraphTraits.h"
#include "llvm/Support/FormatVariadic.h"

#include "revng/ABI/RegisterState.h"
#include "revng/EarlyFunctionAnalysis/CFGAnalyzer.h"
#include "revng/MFP/MFP.h"
#include "revng/MFP/SetLattices.h"
#include "revng/Model/Register.h"
#include "revng/Support/MetaAddress.h"

#include "InterproceduralGraph.h"

namespace efa {

class InterproceduralAnalysis {
public:
  struct LatticeElement {
    using RegisterSet = std::set<const llvm::GlobalVariable *>;
    using SUL = SetUnionLattice<RegisterSet>;

    explicit LatticeElement(bool IsExtremal = false) :
      IsExtremal{ IsExtremal } {}

    RegisterSet Arguments;
    RegisterSet Returns;

    bool IsExtremal = false;
  };

private:
  using CSVSet = std::set<llvm::GlobalVariable *>;
  using ABIMap = std::map<const llvm::GlobalVariable *,
                          abi::RegisterState::Values>;
  using PartialResultsMap = std::map<MetaAddress, ABIAnalyses::ABIAnalysesResults>;

  GeneratedCodeBasicInfo &GCBI;
  TupleTree<model::Binary> &Binary;
  PartialResultsMap PartialResults;

  llvm::Function *EntryHook;
  llvm::Function *PreCallHook;
  llvm::Function *PostCallHook;
  llvm::Function *RetHook;

  InterproceduralGraph InterGraph;

public:
  using Label = efa::InterproceduralNode *;
  using GraphType = efa::InterproceduralNode *;
  using Results = MFP::MFPResult<LatticeElement>;

  explicit InterproceduralAnalysis(GeneratedCodeBasicInfo &GCBI,
                                   TupleTree<model::Binary> &Binary,
                                   PartialResultsMap &PartialResults,
                                   llvm::Function *EntryHook,
                                   llvm::Function *PreCallHook,
                                   llvm::Function *PostCallHook,
                                   llvm::Function *RetHook) :
    GCBI{ GCBI },
    Binary{ Binary },
    PartialResults{ PartialResults },
    EntryHook{ EntryHook },
    PreCallHook{ PreCallHook },
    PostCallHook{ PostCallHook },
    RetHook{ RetHook } {}

  LatticeElement
  combineValues(const LatticeElement &E1, const LatticeElement &E2) const;

  bool isLessOrEqual(const LatticeElement &E1, const LatticeElement &E2) const;

  LatticeElement applyTransferFunction(Label L, const LatticeElement &E2) const;

private:
  void constructInterproceduralGraph();

};

} // namespace efa
