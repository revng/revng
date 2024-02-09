#pragma once

//
// Copyright (c) rev.ng Labs Srl. See LICENSE.md for details.
//

#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instruction.h"

#include "revng/Support/MonotoneFramework.h"

namespace LivenessAnalysis {

using LiveSet = UnionMonotoneSet<const llvm::Instruction *>;
using LivenessMap = std::map<const llvm::BasicBlock *, LiveSet>;

} // end namespace LivenessAnalysis

/// Performs liveness analysis on F. Returns a map with a BasicBlock as key, and
/// the live-in set for that BasicBlock as mapped value.
LivenessAnalysis::LivenessMap computeLiveness(const llvm::Function &F);
