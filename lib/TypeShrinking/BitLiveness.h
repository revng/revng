#pragma once

//
// Copyright rev.ng Srls. See LICENSE.md for details.
//

#include "revng/ADT/GenericGraph.h"

#include "revng-c/TypeShrinking/MFP.h"

#include "DataFlowGraph.h"

namespace llvm {
class Instruction;
} // end namespace llvm

namespace TypeShrinking {

extern const uint32_t Top;

bool isDataFlowSink(const llvm::Instruction *Ins);

/// This class is an instance of monotone framework
/// the elements represent the index from which all bits are not alive
/// so for an element E, all bits with index < E are alive
struct BitLivenessAnalysis : MonotoneFramework<uint32_t,
                                               GenericGraph<DataFlowNode> *,
                                               BitLivenessAnalysis> {
  static uint32_t combineValues(const uint32_t &lh, const uint32_t &rh);
  static uint32_t applyTransferFunction(DataFlowNode *L, const uint32_t E);
  static bool isLessOrEqual(const uint32_t &lh, const uint32_t &rh);
};

} // namespace TypeShrinking
