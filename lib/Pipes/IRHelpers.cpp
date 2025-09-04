/// \file IRHelpers.cpp

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/IR/DebugInfoMetadata.h"
#include "llvm/IR/DebugLoc.h"
#include "llvm/IR/Instruction.h"

#include "revng/Pipes/IRHelpers.h"
#include "revng/Support/IRHelpers.h"

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

llvm::DenseMap<MetaAddress, const llvm::Function *>
getTargetToFunctionMapping(const llvm::Module &M) {
  llvm::DenseMap<MetaAddress, const llvm::Function *> Map;
  for (const llvm::Function &F : M.functions()) {
    auto MA = getMetaAddressMetadata(&F, FunctionEntryMDName);
    if (MA.isValid()) {
      auto [Iterator, Inserted] = Map.try_emplace(MA, &F);
      revng_assert(Inserted);
    }
  }
  return Map;
}

llvm::Error isDebugLocationInvalid(const llvm::DebugLoc &DebugLocation) {
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

llvm::Error isDebugLocationInvalid(const llvm::Instruction &Instruction) {
  return isDebugLocationInvalid(Instruction.getDebugLoc());
}
