#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/ADT/GenericGraph.h"
#include "revng/EarlyFunctionAnalysis/FunctionMetadata.h"
#include "revng/Model/Binary.h"
#include "revng/Yield/CrossRelations/RelationDescription.h"
#include "revng/Yield/Graph.h"

/* TUPLE-TREE-YAML
name: CrossRelations
type: struct
fields:
  - name: Relations
    sequence:
      type: SortedVector
      elementType: RelationDescription
TUPLE-TREE-YAML */

#include "revng/Yield/CrossRelations/Generated/Early/CrossRelations.h"

namespace yield::crossrelations {

class CrossRelations : public generated::CrossRelations {
private:
  struct EdgeLabel {
    yield::crossrelations::RelationType::Values Type;
  };
  using Node = BidirectionalNode<std::string, EdgeLabel>;

public:
  CrossRelations() = default;
  CrossRelations(const SortedVector<efa::FunctionMetadata> &Metadata,
                 const model::Binary &Binary);

  GenericGraph<Node, 16, true> toCallGraph() const;
  yield::Graph toYieldGraph() const;
};

} // namespace yield::crossrelations

#include "revng/Yield/CrossRelations/Generated/Late/CrossRelations.h"
