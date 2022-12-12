#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/ADT/SortedVector.h"
#include "revng/Yield/CrossRelations/RelationTarget.h"

/* TUPLE-TREE-YAML

name: RelationDescription
type: struct
fields:
  - name: Location
    type: string
  - name: Related
    sequence:
      type: SortedVector
      elementType: RelationTarget
key:
  - Location

TUPLE-TREE-YAML */

#include "revng/Yield/CrossRelations/Generated/Early/RelationDescription.h"

namespace yield {

class RelationDescription : public generated::RelationDescription {
public:
  using generated::RelationDescription::RelationDescription;
};

} // namespace yield

#include "revng/Yield/CrossRelations/Generated/Late/RelationDescription.h"
