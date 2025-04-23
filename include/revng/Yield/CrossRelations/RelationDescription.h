#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/ADT/SortedVector.h"

#include "revng/Yield/CrossRelations/Generated/Early/RelationDescription.h"

namespace yield::crossrelations {

class RelationDescription : public generated::RelationDescription {
public:
  using generated::RelationDescription::RelationDescription;
};

} // namespace yield::crossrelations

#include "revng/Yield/CrossRelations/Generated/Late/RelationDescription.h"
