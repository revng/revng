#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/Model/Architecture.h"
#include "revng/Model/Binary.h"

inline uint64_t getExplicitPointerSize(const model::Binary &Model) {
  // If the model does not specify architecture, there is no point in emitting
  // anything other than target-native pointer types.
  if (Model.Architecture() == model::Architecture::Invalid)
    return 0;

  uint64_t PointerSize = getPointerSize(Model.Architecture());

  // Currently we hardcode the target pointer size as 8 (64-bit), so there is
  // no reason to emit explicit pointer sizes for binaries with matching size.
  if (PointerSize == 8)
    return 0;

  return PointerSize;
}
