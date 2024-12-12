#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/EarlyFunctionAnalysis/ControlFlowGraphCache.h"
#include "revng/Support/MetaAddress.h"
#include "revng/Support/MetaAddress/YAMLTraits.h"
#include "revng/Support/YAMLTraits.h"
#include "revng/TupleTree/TupleTree.h"

/* TUPLE-TREE-YAML
name: DecoratedFunction
doc: A decorated function meant to be used for EFA tests
type: struct
fields:
  - name: Entry
    type: MetaAddress
  - name: Name
    type: string
  - name: ControlFlowGraph
    type: efa::ControlFlowGraph
  - name: Attributes
    sequence:
      type: MutableSet
      elementType: model::FunctionAttribute::Values
key:
  - Entry
TUPLE-TREE-YAML */

#include "Generated/Early/DecoratedFunction.h"

class revng::DecoratedFunction : public revng::generated::DecoratedFunction {
public:
  using revng::generated::DecoratedFunction::DecoratedFunction;
};

#include "Generated/Late/DecoratedFunction.h"
