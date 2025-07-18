/// \file DebugInfoHelper.cpp
/// Contains the debug information related helpers than have to depends on
/// `revng::ranks` because they use locations

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/IR/DebugInfoMetadata.h"
#include "llvm/IR/Instruction.h"

#include "revng/Pipeline/Location.h"
#include "revng/Pipes/DebugInfoHelpers.h"
#include "revng/Pipes/Ranks.h"
#include "revng/Support/MetaAddress.h"

std::optional<MetaAddress>
revng::tryExtractAddress(const llvm::Instruction &I) {
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
