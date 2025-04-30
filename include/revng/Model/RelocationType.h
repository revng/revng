#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/Model/Architecture.h"

#include "revng/Model/Generated/Early/RelocationType.h"

constexpr unsigned char R_MIPS_IMPLICIT_RELATIVE = 255;

namespace model::RelocationType {

inline uint64_t getSize(Values V) {
  switch (V) {
  case WriteAbsoluteAddress32:
  case AddAbsoluteAddress32:
  case WriteRelativeAddress32:
  case AddRelativeAddress32:
    return 4;

  case WriteAbsoluteAddress64:
  case AddAbsoluteAddress64:
  case WriteRelativeAddress64:
  case AddRelativeAddress64:
    return 8;

  default:
    revng_abort();
  }
}

Values fromELFRelocation(model::Architecture::Values Architecture,
                         unsigned char ELFRelocation);

Values formCOFFRelocation(model::Architecture::Values Architecture);

bool isELFRelocationBaseRelative(model::Architecture::Values Architecture,
                                 unsigned char ELFRelocation);

} // namespace model::RelocationType

#include "revng/Model/Generated/Late/RelocationType.h"
