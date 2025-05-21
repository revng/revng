#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/Pipeline/Location.h"
#include "revng/Pipes/Ranks.h"
#include "revng/Support/MetaAddress.h"

namespace revng {

// TODO: find a better place to put this.

inline std::optional<MetaAddress>
tryExtractAddress(const llvm::Instruction &I) {
  if (!I.getDebugLoc() || !I.getDebugLoc()->getScope())
    return std::nullopt;

  auto DebugLocation = I.getDebugLoc()->getScope()->getName().str();
  auto Parsed = pipeline::locationFromString(revng::ranks::Instruction,
                                             DebugLocation);
  revng_assert(Parsed.has_value());

  MetaAddress Extracted = Parsed->at(revng::ranks::Instruction);
  revng_assert(Extracted.isValid());
  return Extracted;
}

} // namespace revng
