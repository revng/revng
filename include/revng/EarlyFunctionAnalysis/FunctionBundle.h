#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/EarlyFunctionAnalysis/ControlFlowGraph.h"
#include "revng/Support/YAMLTraits.h"
#include "revng/TupleTree/TupleTree.h"

#include "revng/EarlyFunctionAnalysis/Generated/Early/FunctionBundle.h"

class efa::FunctionBundle : public efa::generated::FunctionBundle {
public:
  using generated::FunctionBundle::FunctionBundle;

public:
  /// \return the block \p I belongs to and the control-flow graph describing
  ///         it, or a pair of `nullptr` if there is no such block
  std::pair<const efa::ControlFlowGraph *, const efa::BasicBlock *>
  findBlock(llvm::Instruction *I) const;
};

#include "revng/EarlyFunctionAnalysis/Generated/Late/FunctionBundle.h"
