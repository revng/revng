/// \file IRHelpers.cpp
/// Implementation of IR helper functions.

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/IR/DebugInfoMetadata.h"
#include "llvm/IR/DebugLoc.h"
#include "llvm/IR/Instruction.h"

#include "revng/EarlyFunctionAnalysis/IRHelpers.h"
#include "revng/Pipeline/Location.h"
#include "revng/Pipes/Ranks.h"

// Unless your helpers need to depend on `revngPipeline` (for example, if they
// create/parse `Location`s), you should place them in
// `include/revng/Support/IRHelpers.h` instead.

llvm::Error checkDebugLocationValidity(const llvm::DebugLoc &DebugLocation) {
  if (not DebugLocation)
    return revng::createError("the debug location is empty");

  if (not DebugLocation->getScope())
    return revng::createError("the debug location has no scope component");

  //
  // TODO: don't forget to update this when we add more structure to debug
  //       information we attach (for example, when we allow for more than one
  //       address).
  //

  const auto &Serialized = DebugLocation->getScope()->getName();
  if (Serialized.empty())
    return revng::createError("the scope component has an empty name");

  // This check is kind of expensive, we might want it hidden away behind
  // `if (VerifyLog.isEnabled())` in the general case because of how unlikely
  // it is to trigger.
  if (not pipeline::locationFromString(revng::ranks::Instruction,
                                       Serialized.str()))
    return revng::createError("the scope component is not a valid instruction "
                              "location");

  return llvm::Error::success();
}

llvm::Error checkDebugLocationValidity(const llvm::Instruction &Instruction) {
  return checkDebugLocationValidity(Instruction.getDebugLoc());
}
