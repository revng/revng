#pragma once

//
// This file is distributed under the MIT License. See LICENSE.mit for details.
//

#include "revng/ADT/SortedVector.h"

/* TUPLE-TREE-YAML

name: RelationDescription
type: struct
fields:
  - name: Location
    type: string
  - name: IsCalledFrom
    sequence:
      type: SortedVector
      elementType: string
    optional: true
key:
  - Location

TUPLE-TREE-YAML */

#include "revng/Yield/CrossRelations/Generated/Early/RelationDescription.h"

namespace yield::crossrelations {

class RelationDescription : public generated::RelationDescription {
public:
  using generated::RelationDescription::RelationDescription;
};

} // namespace yield::crossrelations

#include "revng/Yield/CrossRelations/Generated/Late/RelationDescription.h"
