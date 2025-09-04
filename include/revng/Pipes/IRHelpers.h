#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <optional>

#include "llvm/ADT/DenseMap.h"

#include "revng/Pipeline/Location.h"
#include "revng/Pipes/Ranks.h"
#include "revng/Support/MetaAddress.h"

std::optional<pipeline::Location<decltype(revng::ranks::Instruction)>>
getLocation(const llvm::Instruction *I);

[[nodiscard]] llvm::DenseMap<MetaAddress, const llvm::Function *>
getTargetToFunctionMapping(const llvm::Module &M);

namespace llvm {
class DebugLoc;
class Instruction;
} // namespace llvm

// This ensures debug information validity.
//
// In revng modules, valid debug information location is one that is:
// - non-empty (`!!DebugLoc`),
// - has a scope (`DebugLoc->getScope()`) with a non-empty name,
// - where the name is a valid `/instruction/...` location.
// (the last one is subject to change when we start attaching more than one
//  address per llvm instruction).
llvm::Error isDebugLocationInvalid(const llvm::DebugLoc &Instruction);
llvm::Error isDebugLocationInvalid(const llvm::Instruction &Instruction);
