#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <optional>

#include "llvm/ADT/DenseMap.h"

#include "revng/Ranks/Location.h"
#include "revng/Ranks/Ranks.h"
#include "revng/Support/MetaAddress.h"

[[nodiscard]] llvm::DenseMap<MetaAddress, const llvm::Function *>
getTargetToFunctionMapping(const llvm::Module &M);

namespace llvm {
class DebugLoc;
class Instruction;
} // namespace llvm
