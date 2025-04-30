#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <cstdint>

#include "revng/ABI/Generated/Early/ScalarType.h"

namespace abi {

class ScalarType : public generated::ScalarType {
public:
  using generated::ScalarType::ScalarType;

  uint64_t alignedAt() const {
    revng_assert(Size() != 0);
    return AlignedAt() != 0 ? AlignedAt() : Size();
  }
};

} // namespace abi

#include "revng/ABI/Generated/Late/ScalarType.h"
