#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/ADT/SortedVector.h"
#include "revng/EarlyFunctionAnalysis/CallEdge.h"
#include "revng/EarlyFunctionAnalysis/FunctionEdge.h"
#include "revng/EarlyFunctionAnalysis/FunctionEdgeBase.h"
#include "revng/Model/Identifier.h"
#include "revng/Model/VerifyHelper.h"
#include "revng/Support/BasicBlockID.h"
#include "revng/Support/BasicBlockID/YAMLTraits.h"
#include "revng/Support/MetaAddress.h"

/* TUPLE-TREE-YAML
name: BasicBlock
doc: The basic block of a function
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
key:
  - ID
TUPLE-TREE-YAML */

#include "revng/EarlyFunctionAnalysis/Generated/Early/BasicBlock.h"

class efa::BasicBlock : public efa::generated::BasicBlock {
public:
  using generated::BasicBlock::BasicBlock;

public:
  model::Identifier name() const;
  BasicBlockID nextBlock() const {
    return BasicBlockID(End(), ID().inliningIndex());
  }

  [[nodiscard]] bool contains(const BasicBlockID &Other) const {
    if (Other.inliningIndex() != ID().inliningIndex())
      return false;

    MetaAddress Target = Other.start();
    MetaAddress Start = ID().start();
    MetaAddress End = this->End();
    return Start <= Target and Target < End;
  }

public:
  bool verify() const debug_function;
  bool verify(bool Assert) const debug_function;
  bool verify(model::VerifyHelper &VH) const;
  void dump() const debug_function;
};

inline model::TypePath getPrototype(const model::Binary &Binary,
                                    MetaAddress CallerFunctionAddress,
                                    const efa::BasicBlock &CallerBlock,
                                    const efa::CallEdge &Edge) {
  model::TypePath Result;

  MetaAddress Caller = CallerFunctionAddress;
  Caller = CallerBlock.InlinedFrom();

  auto &CallSitePrototypes = Binary.Functions()
                               .at(CallerFunctionAddress)
                               .CallSitePrototypes();
  auto It = CallSitePrototypes.find(CallerBlock.ID().start());
  if (It != CallSitePrototypes.end())
    Result = It->Prototype();

  if (Edge.Type() == efa::FunctionEdgeType::FunctionCall) {
    if (not Edge.DynamicFunction().empty()) {
      // Get the dynamic function prototype
      Result = Binary.ImportedDynamicFunctions()
                 .at(Edge.DynamicFunction())
                 .Prototype();
    } else if (Edge.Destination().isValid()) {
      // Get the function prototype
      Result = Binary.Functions().at(Edge.Destination().start()).Prototype();
    }
  }

  if (not Result.isValid())
    Result = Binary.DefaultPrototype();

  return Result;
}

#include "revng/EarlyFunctionAnalysis/Generated/Late/BasicBlock.h"
