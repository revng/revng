#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/EarlyFunctionAnalysis/CallEdge.h"
#include "revng/EarlyFunctionAnalysis/FunctionEdge.h"
#include "revng/EarlyFunctionAnalysis/FunctionEdgeBase.h"
#include "revng/Model/Identifier.h"
#include "revng/Model/VerifyHelper.h"
#include "revng/Support/MetaAddress.h"

/* TUPLE-TREE-YAML
name: BasicBlock
doc: The basic block of a function
type: struct
fields:
  - name: Start
    doc: Start address of the basic block
    type: MetaAddress
  - name: End
    doc: |
      End address of the basic block, i.e., the address where the last
      instruction ends
    type: MetaAddress
  - name: Successors
    doc: List of successor edges
    sequence:
      type: SortedVector
      upcastable: true
      elementType: efa::FunctionEdgeBase
key:
  - Start
TUPLE-TREE-YAML */

#include "revng/EarlyFunctionAnalysis/Generated/Early/BasicBlock.h"

class efa::BasicBlock : public efa::generated::BasicBlock {
public:
  using generated::BasicBlock::BasicBlock;

public:
  model::Identifier name() const;

public:
  bool verify() const debug_function;
  bool verify(bool Assert) const debug_function;
  bool verify(model::VerifyHelper &VH) const;
  void dump() const debug_function;
};

#include "revng/EarlyFunctionAnalysis/Generated/Late/BasicBlock.h"
