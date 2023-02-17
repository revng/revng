#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <limits>
#include <string>

#include "revng/ADT/SortedVector.h"
#include "revng/Model/VerifyHelper.h"
#include "revng/Support/MetaAddress.h"
#include "revng/Support/MetaAddress/YAMLTraits.h"
#include "revng/Yield/CallEdge.h"
#include "revng/Yield/FunctionEdge.h"
#include "revng/Yield/FunctionEdgeBase.h"
#include "revng/Yield/Instruction.h"

/* TUPLE-TREE-YAML

name: BasicBlock
type: struct
fields:
  - name: ID
    type: BasicBlockID

  - name: End
    doc: |
      End address of the basic block, i.e., the address where the last
      instruction ends
    type: MetaAddress

  - name: InlinedFrom
    type: MetaAddress
    doc: Address of the function this basic block has been inlined from

  - name: Successors
    doc: List of successor edges
    sequence:
      type: SortedVector
      upcastable: true
      elementType: FunctionEdgeBase

  - name: Instructions
    sequence:
      type: SortedVector
      elementType: Instruction

  - name: IsLabelAlwaysRequired
    doc: |
      This flag is set to `false` for basic blocks that are never directly
      pointed to, i.e. blocks that are only ever entered from the previous
      instructions and such.
      This lets us dynamically decide whether we want to show labels like this
      or not.
    type: bool

  - name: HasDelaySlot
    doc: >
      This flag is set if the last instruction of the block is in a delay slot
      and is executed at the same time as the instruction preceding it.
      \note: This is always equal to `false` on architectures that do not
      support delay slots.
    type: bool
    optional: true

key:
  - ID

TUPLE-TREE-YAML */

#include "revng/Yield/Generated/Early/BasicBlock.h"

namespace yield {

class BasicBlock : public generated::BasicBlock {
public:
  using generated::BasicBlock::BasicBlock;

public:
  BasicBlockID nextBlock() const {
    return BasicBlockID(End(), ID().inliningIndex());
  }

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
