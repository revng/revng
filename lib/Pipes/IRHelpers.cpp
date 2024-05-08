/// \file IRHelpers.cpp

//
// This file is distributed under the MIT License. See LICENSE.mit for details.
//

#include "llvm/IR/DebugInfoMetadata.h"
#include "llvm/IR/Instruction.h"

#include "revng/Pipes/IRHelpers.h"

std::optional<pipeline::Location<decltype(revng::ranks::Instruction)>>
getLocation(const llvm::Instruction *I) {
  auto MaybeDebugLoc = I->getDebugLoc();
  if (not MaybeDebugLoc or MaybeDebugLoc.getInlinedAt() == nullptr)
    return std::nullopt;

  using Location = pipeline::Location<decltype(revng::ranks::Instruction)>;
  auto Result = Location::fromString(MaybeDebugLoc->getScope()->getName());
  revng_assert(Result);

  return Result;
}
