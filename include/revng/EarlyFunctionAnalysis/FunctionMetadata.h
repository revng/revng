#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/EarlyFunctionAnalysis/BasicBlock.h"
#include "revng/Model/VerifyHelper.h"
#include "revng/Support/MetaAddress.h"
#include "revng/Support/MetaAddress/YAMLTraits.h"
#include "revng/Support/YAMLTraits.h"
#include "revng/TupleTree/TupleTree.h"

/* TUPLE-TREE-YAML
name: FunctionMetadata
doc: "Metadata attached to a function. As of now, it includes a list of basic
blocks, representing the control-flow graph. It is meant to include further
information in the future."
type: struct
fields:
  - name: Entry
    doc: Start address of the basic block
    type: MetaAddress
  - name: ControlFlowGraph
    sequence:
      type: SortedVector
      elementType: efa::BasicBlock
    optional: true
key:
  - Entry
TUPLE-TREE-YAML */

#include "revng/EarlyFunctionAnalysis/Generated/Early/FunctionMetadata.h"

class efa::FunctionMetadata : public efa::generated::FunctionMetadata {
public:
  using generated::FunctionMetadata::FunctionMetadata;

public:
  bool verify(const model::Binary &Binary) const debug_function;
  bool verify(const model::Binary &Binary, bool Assert) const debug_function;
  bool verify(const model::Binary &Binary, model::VerifyHelper &VH) const;
  void dump() const debug_function;
  void dumpCFG(const model::Binary &Binary) const debug_function;
};

#include "revng/EarlyFunctionAnalysis/Generated/Late/FunctionMetadata.h"
