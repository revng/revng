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


struct ABI {
  using RegisterSet = std::set<const llvm::GlobalVariable *>;
  using SUL = SetUnionLattice<RegisterSet>;

  RegisterSet Arguments;
  RegisterSet Returns;

  bool isLessOrEqual(const ABI &RHS) const;

  ABI combineValues(const ABI &) const;
};

struct InterproceduralLattice {
  explicit InterproceduralLattice(bool IsExtremal = false) :
    IsExtremal{ IsExtremal } {}

  std::map<Node::AddressType, ABI> FinalABI;

  bool IsExtremal = false;
};

class InterproceduralAnalysis {
public:

private:
  using CSVSet = std::set<llvm::GlobalVariable *>;
  using ABIMap = std::map<const llvm::GlobalVariable *,
                          abi::RegisterState::Values>;
  // TODO: First save all results to FinalResults which uses same keys as
  // interprocedural graph. Then traverse graph and apply changes to
  // FinalResults. Each time some partial results are needed, fetch them from
  // FinalResults.
  //
  // What with RetHook? 
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
  using Results = MFP::MFPResult<InterproceduralLattice>;
  using LatticeElement = efa::InterproceduralLattice;

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

  InterproceduralLattice
  combineValues(const InterproceduralLattice &E1, const InterproceduralLattice &E2) const;

  bool isLessOrEqual(const InterproceduralLattice &E1, const InterproceduralLattice &E2) const;

  InterproceduralLattice applyTransferFunction(Label L, const InterproceduralLattice &E2) const;

private:
  void constructInterproceduralGraph();

  static void copyResults(const ABIAnalyses::RegisterStateMap &Source,
                          ABI::RegisterSet &Destination) {
    for (const auto &[CSV, State] : Source) {
      if (State == abi::RegisterState::Yes) {
        Destination.insert(CSV);
      }
    }
  }
};

} // namespace efa
