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

  uint64_t BinaryPointerSize = getPointerSize(Model.Architecture());
  uint64_t TargetPointerSize = getPointerSize(getArchitecture(Model
                                                                .targetABI()));

  // If the binary and target pointer sizes match, there is no need to emit
  // explicit pointer sizes.
  if (BinaryPointerSize == TargetPointerSize)
    return 0;

  return TargetPointerSize;
}
