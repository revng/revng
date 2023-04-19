#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <optional>

#include "revng/Pipeline/Location.h"
#include "revng/Pipes/Ranks.h"

std::optional<pipeline::Location<decltype(revng::ranks::Instruction)>>
getLocation(const llvm::Instruction *I);
