#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/Yield/CrossRelations/RelationType.h"

/* TUPLE-TREE-YAML

name: RelationTarget
type: struct
fields:
  - name: Kind
    type: RelationType
  - name: Location
    type: string
key:
  - Kind
  - Location

TUPLE-TREE-YAML */

#include "revng/Yield/CrossRelations/Generated/Early/RelationTarget.h"

namespace yield::crossrelations {

class RelationTarget : public generated::RelationTarget {
public:
  using generated::RelationTarget::RelationTarget;
};

} // namespace yield::crossrelations

#include "revng/Yield/CrossRelations/Generated/Late/RelationTarget.h"
