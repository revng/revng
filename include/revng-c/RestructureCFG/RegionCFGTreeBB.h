#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <cstdlib>
#include <set>

// Forward declarations
namespace llvm {
class BasicBlock;
} // namespace llvm

#include "revng-c/RestructureCFG/BasicBlockNodeBB.h"
#include "revng-c/RestructureCFG/RegionCFGTree.h"
extern template class RegionCFG<llvm::BasicBlock *>;

using RegionCFGBB = RegionCFG<llvm::BasicBlock *>;
