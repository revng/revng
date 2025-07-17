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
