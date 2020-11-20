#pragma once

//
// Copyright rev.ng Srls. See LICENSE.md for details.
//

#include "revng-c/TypeShrinking/MFP.h"

#include "DataFlowGraph.h"

namespace llvm {
class Instruction;
} // end namespace llvm

namespace TypeShrinking {
bool hasSideEffect(const llvm::Instruction *Ins);

/// This class is an instance of monotone framework
/// the elements represent the index from which all bits are not alive
/// so for an element E, all bits with index < E are alive
struct BitLivenessAnalysis : MonotoneFramework<unsigned,
                                               GenericGraph<DataFlowNode> *,
                                               BitLivenessAnalysis> {
  static unsigned combineValues(const unsigned &lh, const unsigned &rh);
  static unsigned applyTransferFunction(DataFlowNode *L, const unsigned E);
  static bool isLessOrEqual(const unsigned &lh, const unsigned &rh);
};

} // namespace TypeShrinking
