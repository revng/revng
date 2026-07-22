#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <cstdint>

#include "revng/Model/ABI/CType.h"
#include "revng/Model/ABI/ScalarKind.h"

#include "revng/Model/ABI/Generated/Early/ScalarType.h"

namespace abi {

class ScalarType : public abi::generated::ScalarType {
public:
  using abi::generated::ScalarType::ScalarType;

  uint64_t alignedAt() const {
    revng_assert(Size() != 0);
    return AlignedAt() != 0 ? AlignedAt() : Size();
  }
};

} // namespace abi

#include "revng/Model/ABI/Generated/Late/ScalarType.h"
