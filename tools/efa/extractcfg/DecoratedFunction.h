#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/EarlyFunctionAnalysis/IRHelpers.h"
#include "revng/Support/MetaAddress.h"
#include "revng/Support/MetaAddress/YAMLTraits.h"
#include "revng/Support/YAMLTraits.h"
#include "revng/TupleTree/TupleTree.h"

/* TUPLE-TREE-YAML
name: DecoratedFunction
doc: A decorated function meant to be used for EFA tests
type: struct
fields:
  - name: OriginalName
    type: std::string
  - name: Type
    doc: Type of the function
    type: model::FunctionType::Values
  - name: FunctionMetadata
    type: efa::FunctionMetadata
key:
  - OriginalName
TUPLE-TREE-YAML */

#include "Generated/Early/DecoratedFunction.h"

class revng::DecoratedFunction : public revng::generated::DecoratedFunction {
public:
  using revng::generated::DecoratedFunction::DecoratedFunction;
};

#include "Generated/Late/DecoratedFunction.h"
