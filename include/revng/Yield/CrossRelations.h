#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/ADT/GenericGraph.h"
#include "revng/EarlyFunctionAnalysis/FunctionMetadata.h"
#include "revng/Model/Binary.h"
#include "revng/Yield/Graph.h"
#include "revng/Yield/RelationDescription.h"

/* TUPLE-TREE-YAML
name: CrossRelations
type: struct
fields:
  - name: Relations
    sequence:
      type: SortedVector
      elementType: RelationDescription
TUPLE-TREE-YAML */

#include "revng/Yield/Generated/Early/CrossRelations.h"

namespace yield {

class CrossRelations : public generated::CrossRelations {
private:
  struct EdgeLabel {
    yield::RelationType::Values Type;
  };
  using Node = BidirectionalNode<std::string, EdgeLabel>;

public:
  CrossRelations() = default;
  CrossRelations(const SortedVector<efa::FunctionMetadata> &Metadata,
                 const model::Binary &Binary);

  GenericGraph<Node, 16, true> toCallGraph() const;
  yield::Graph toYieldGraph() const;
};

} // namespace yield

#include "revng/Yield/Generated/Late/CrossRelations.h"
