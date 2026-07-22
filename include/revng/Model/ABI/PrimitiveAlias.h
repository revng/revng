#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <cstdint>

#include "revng/Model/PrimitiveKind.h"

#include "revng/Model/ABI/Generated/Early/PrimitiveAlias.h"

namespace abi {

class PrimitiveAlias : public generated::PrimitiveAlias {
public:
  using generated::PrimitiveAlias::PrimitiveAlias;
};

} // namespace abi

#include "revng/Model/ABI/Generated/Late/PrimitiveAlias.h"
