#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/ADT/SortedVector.h"
#include "revng/Support/MetaAddress.h"

#include "revng/Yield/CrossRelations/Generated/Early/GlobalDescription.h"

namespace yield::crossrelations {

class GlobalDescription : public generated::GlobalDescription {
public:
  using generated::GlobalDescription::GlobalDescription;

  /// Whether this global variable holds a string.
  bool isString() const { return Encoding() != StringEncoding::Invalid; }
};

} // namespace yield::crossrelations

#include "revng/Yield/CrossRelations/Generated/Late/GlobalDescription.h"
