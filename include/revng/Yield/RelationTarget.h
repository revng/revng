#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/Yield/RelationType.h"

/* TUPLE-TREE-YAML

name: RelationTarget
type: struct
fields:
  - name: Kind
    type: yield::RelationType::Values
  - name: Location
    type: std::string
key:
  - Kind
  - Location

TUPLE-TREE-YAML */

#include "revng/Yield/Generated/Early/RelationTarget.h"

namespace yield {

class RelationTarget : public generated::RelationTarget {
public:
  using generated::RelationTarget::RelationTarget;
};

} // namespace yield

#include "revng/Yield/Generated/Late/RelationTarget.h"
