#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <utility>

#include "revng/Model/Architecture.h"

#include "revng/Model/Generated/Early/RelocationType.h"

constexpr unsigned char R_MIPS_IMPLICIT_RELATIVE = 255;

namespace model::RelocationType {

/// \return the type and the size of the relocation corresponding to
///         \p ELFRelocation, or `Invalid` if it's not supported
std::pair<Values, uint64_t>
fromELFRelocation(model::Architecture::Values Architecture,
                  unsigned char ELFRelocation);

/// \return the type and the size of the relocation of an entry of the import
///         address table
std::pair<Values, uint64_t>
formCOFFRelocation(model::Architecture::Values Architecture);

bool isELFRelocationBaseRelative(model::Architecture::Values Architecture,
                                 unsigned char ELFRelocation);

} // namespace model::RelocationType

#include "revng/Model/Generated/Late/RelocationType.h"
