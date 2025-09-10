#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/Support/Error.h"

// Unless your helpers need to depend on `revngPipeline` (for example, if they
// create/parse `pipeline::Location`s), you should place them in
// `include/revng/Support/IRHelpers.h` instead.

namespace llvm {
class DebugLoc;
class Instruction;
} // namespace llvm

llvm::Error checkDebugLocationValidity(const llvm::DebugLoc &Instruction);
llvm::Error checkDebugLocationValidity(const llvm::Instruction &Instruction);
