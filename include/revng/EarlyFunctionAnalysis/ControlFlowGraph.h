#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/EarlyFunctionAnalysis/BasicBlock.h"
#include "revng/Support/MetaAddress.h"
#include "revng/Support/MetaAddress/YAMLTraits.h"
#include "revng/Support/YAMLTraits.h"
#include "revng/TupleTree/TupleTree.h"

namespace llvm {
class BasicBlock;
}
class GeneratedCodeBasicInfo;

/* TUPLE-TREE-YAML
name: ControlFlowGraph
doc: "Metadata attached to a function. As of now, it includes a list of basic
blocks, representing the control-flow graph."
type: struct
fields:
  - name: Entry
    doc: Start address of the basic block
    type: MetaAddress
  - name: Name
    type: string
    doc: Optional name for debugging purposes
    optional: true
  - name: Blocks
    sequence:
      type: SortedVector
      elementType: BasicBlock
    optional: true
key:
  - Entry
TUPLE-TREE-YAML */

#include "revng/EarlyFunctionAnalysis/Generated/Early/ControlFlowGraph.h"

namespace model {
class VerifyHelper;
}

class efa::ControlFlowGraph : public efa::generated::ControlFlowGraph {
public:
  using generated::ControlFlowGraph::ControlFlowGraph;

public:
  const efa::BasicBlock *findBlock(GeneratedCodeBasicInfo &GCBI,
                                   llvm::BasicBlock *BB) const;

  void serialize(GeneratedCodeBasicInfo &GCBI) const;

public:
  bool verify(const model::Binary &Binary) const debug_function;
  bool verify(const model::Binary &Binary, bool Assert) const debug_function;
  bool verify(const model::Binary &Binary, model::VerifyHelper &VH) const;
  void dumpCFG(const model::Binary &Binary) const debug_function;

public:
  void simplify(const model::Binary &Binary);
};

#include "revng/EarlyFunctionAnalysis/Generated/Late/ControlFlowGraph.h"
