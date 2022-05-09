#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <limits>
#include <string>

#include "revng/ADT/SortedVector.h"
#include "revng/EarlyFunctionAnalysis/BasicBlock.h"
#include "revng/Model/VerifyHelper.h"
#include "revng/Support/MetaAddress.h"
#include "revng/Support/MetaAddress/YAMLTraits.h"
#include "revng/Yield/Instruction.h"

/* TUPLE-TREE-YAML

name: BasicBlock
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

  - name: Instructions
    sequence:
      type: SortedVector
      elementType: yield::Instruction

  - name: IsLabelAlwaysRequired
    doc: |
      This flag is set to `false` for basic blocks that are never directly
      pointed to, i.e. blocks that are only ever entered from the previous
      instructions and such.
      This lets us dynamically decide whether we want to show labels like this
      or not.
    type: bool

key:
  - Start

TUPLE-TREE-YAML */

#include "revng/Yield/Generated/Early/BasicBlock.h"

namespace yield {

class BasicBlock : public generated::BasicBlock {
public:
  using generated::BasicBlock::BasicBlock;

public:
  bool verify(model::VerifyHelper &VH) const;
  void dump() const debug_function;

public:
  inline bool verify() const debug_function { return verify(false); }
  inline bool verify(bool Assert) const debug_function {
    model::VerifyHelper VH(Assert);
    return verify(VH);
  }
};

} // namespace yield

#include "revng/Yield/Generated/Late/BasicBlock.h"
