#pragma once

//
// This file is distributed under the MIT License. See LICENSE.mit for details.
//

#include <map>

#include "llvm/Analysis/LazyValueInfo.h"
#include "llvm/Support/GenericDomTree.h"

#include "revng/ADT/ConstantRangeSet.h"
#include "revng/MFP/MFP.h"
#include "revng/Support/Statistics.h"
#include "revng/ValueMaterializer/AdvancedValueInfo.h"
#include "revng/ValueMaterializer/ControlFlowEdgesGraph.h"
#include "revng/ValueMaterializer/DataFlowGraph.h"
#include "revng/ValueMaterializer/MemoryOracle.h"

namespace llvm {
class LazyValueInfo;
class DominatorTree;
} // namespace llvm

namespace Oracle {
enum Values {
  None,
  LazyValueInfo,
  AdvancedValueInfo,
  Count
};
} // namespace Oracle

inline RunningStatistics DFGSizeStatitistics("vm-dfg-size");

class ValueMaterializer {
private:
  using ConstraintsMap = std::map<llvm::Instruction *, ConstantRangeSet>;

private:
  //
  // Inputs
  //
  llvm::Instruction *Context;
  llvm::Value *V;
  MemoryOracle &MO;
  llvm::LazyValueInfo &LVI;
  const llvm::DominatorTree &DT;
  DataFlowGraph::Limits TheLimits;
  Oracle::Values Oracle;

  //
  // Outputs
  //
  DataFlowGraph DataFlowGraph;
  ConstraintsMap OracleConstraints;
  map<const ForwardNode<ControlFlowEdgesNode> *,
      MFP::MFPResult<map<llvm::Instruction *, ConstantRangeSet>>>
    MFIResults;
  std::optional<MaterializedValues> Values;
  ControlFlowEdgesGraph CFEG;

private:
  ValueMaterializer(llvm::Instruction *Context,
                    llvm::Value *V,
                    MemoryOracle &MO,
                    llvm::LazyValueInfo &LVI,
                    const llvm::DominatorTree &DT,
                    DataFlowGraph::Limits TheLimits,
                    Oracle::Values Oracle) :
    Context(Context),
    V(V),
    MO(MO),
    LVI(LVI),
    DT(DT),
    TheLimits(TheLimits),
    Oracle(Oracle) {}

public:
  static ValueMaterializer getValuesFor(llvm::Instruction *Context,
                                        llvm::Value *V,
                                        MemoryOracle &MO,
                                        llvm::LazyValueInfo &LVI,
                                        const llvm::DominatorTree &DT,
                                        DataFlowGraph::Limits TheLimits,
                                        Oracle::Values Oracle) {
    ValueMaterializer Result(Context, V, MO, LVI, DT, TheLimits, Oracle);
    Result.run();
    return Result;
  }

public:
  const auto &dataFlowGraph() const { return DataFlowGraph; }
  const auto &oracleConstraints() const { return OracleConstraints; }
  const auto &mfiResult() const { return MFIResults; }
  const auto &values() const { return Values; }
  const auto &cfeg() const { return CFEG; }

private:
  void run();

  void computeOracleConstraints();

  void applyOracleResultsToDataFlowGraph();

  void computeSizeLowerBound();

  void electMaterializationStartingPoints();
};
