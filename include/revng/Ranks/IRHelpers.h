#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/IR/DebugInfoMetadata.h"
#include "llvm/IR/Instruction.h"

#include "revng/Ranks/Location.h"
#include "revng/Ranks/Ranks.h"

using InstructionLocation = pipeline::Location<
  decltype(revng::ranks::Instruction)>;

std::optional<InstructionLocation> inline getLocation(const llvm::Instruction
                                                        *I) {
  auto MaybeDebugLoc = I->getDebugLoc();
  if (not MaybeDebugLoc or MaybeDebugLoc.getInlinedAt() == nullptr)
    return std::nullopt;

  using Location = pipeline::Location<decltype(revng::ranks::Instruction)>;
  auto Result = Location::fromString(MaybeDebugLoc->getScope()->getName());
  revng_assert(Result);

  return Result;
}

// This ensures debug information validity.
//
// In revng modules, valid debug information location is one that is:
// - non-empty (`bool(DebugLoc)`),
// - has a scope (`DebugLoc->getScope()`) with a non-empty name,
// - where the name is a valid `/instruction/...` location.
// (the last one is subject to change when we start attaching more than one
//  address per llvm instruction).
inline llvm::Error isDebugLocationInvalid(const llvm::DebugLoc &DebugLocation) {
  if (not DebugLocation)
    return revng::createError("The debug location is empty.");

  if (not DebugLocation->getScope())
    return revng::createError("The debug location has no scope component.");

  //
  // TODO: don't forget to update this when we add more structure to debug
  //       information we attach (for example, when we allow for more than one
  //       address).
  //

  const auto &Serialized = DebugLocation->getScope()->getName();
  if (Serialized.empty())
    return revng::createError("The scope component has an empty name.");

  // This check is kind of expensive, we might want it hidden away behind
  // `if (VerifyLog.isEnabled())` in the general case because of how unlikely
  // it is to ever trigger.
  if (not pipeline::locationFromString(revng::ranks::Instruction,
                                       Serialized.str()))
    return revng::createError("The scope component name is not a valid "
                              "instruction location.");

  return llvm::Error::success();
}

inline llvm::Error
isDebugLocationInvalid(const llvm::Instruction &Instruction) {
  return isDebugLocationInvalid(Instruction.getDebugLoc());
}

std::optional<MetaAddress> inline tryExtractAddress(const llvm::Instruction
                                                      &I) {
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
