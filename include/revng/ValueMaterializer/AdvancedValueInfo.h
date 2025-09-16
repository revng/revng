#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <map>

#include "revng/ADT/ConstantRangeSet.h"
#include "revng/MFP/Graph.h"
#include "revng/MFP/MFP.h"
#include "revng/ValueMaterializer/ControlFlowEdgesGraph.h"
#include "revng/ValueMaterializer/DataFlowRangeAnalysis.h"

namespace llvm {
class Instruction;
class LazyValueInfo;
class DominatorTree;
} // namespace llvm

class DataFlowGraph;

class AdvancedValueInfoMFI {
public:
  using LatticeElement = std::map<llvm::Instruction *, ConstantRangeSet>;
  using GraphType = const ControlFlowEdgesGraph *;
  using Label = const ControlFlowEdgesGraph::Node *;
  using ResultsMap = std::map<Label, MFP::MFPResult<LatticeElement>>;
  using InstructionsSet = llvm::SmallPtrSetImpl<llvm::Instruction *>;

private:
  llvm::LazyValueInfo &LVI;
  DataFlowRangeAnalysis &DFRA;
  const llvm::DominatorTree &DT;
  llvm::Instruction *Context;
  llvm::SmallPtrSetImpl<llvm::Instruction *> &Instructions;
  bool ZeroExtendConstraints = false;

public:
  AdvancedValueInfoMFI(llvm::LazyValueInfo &LVI,
                       DataFlowRangeAnalysis &DFRA,
                       const llvm::DominatorTree &DT,
                       llvm::Instruction *Context,
                       InstructionsSet &Instructions,
                       bool ZeroExtendConstraints) :
    LVI(LVI),
    DFRA(DFRA),
    DT(DT),
    Context(Context),
    Instructions(Instructions),
    ZeroExtendConstraints(ZeroExtendConstraints) {}

public:
  LatticeElement combineValues(const LatticeElement &LHS,
                               const LatticeElement &RHS) const;

  bool isLessOrEqual(const LatticeElement &LHS,
                     const LatticeElement &RHS) const;

  LatticeElement applyTransferFunction(Label L, const LatticeElement &E) const;

public:
  static void dump(GraphType CFEG, const ResultsMap &AllResults);
};

static_assert(MFP::MonotoneFrameworkInstance<AdvancedValueInfoMFI>);

/// \p DFG the data flow graph containing the instructions we're interested in.
/// \p Context the position in the function for the current query.
std::tuple<std::map<llvm::Instruction *, ConstantRangeSet>,
           ControlFlowEdgesGraph,
           map<const ForwardNode<ControlFlowEdgesNode> *,
               MFP::MFPResult<map<llvm::Instruction *, ConstantRangeSet>>>>
runAVI(const DataFlowGraph &DFG,
       llvm::Instruction *Context,
       const llvm::DominatorTree &DT,
       llvm::LazyValueInfo &LVI,
       DataFlowRangeAnalysis &DFRA,
       bool ZeroExtendConstraints);

template<>
void MFP::dump(llvm::raw_ostream &Stream,
               unsigned Indent,
               const std::map<llvm::Instruction *, ConstantRangeSet> &Element);
