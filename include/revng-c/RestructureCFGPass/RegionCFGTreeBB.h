#ifndef REVNGC_RESTRUCTURE_CFG_REGIONCFGTREEBB_H
#define REVNGC_RESTRUCTURE_CFG_REGIONCFGTREEBB_H

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// Standard includes
#include <cstdlib>
#include <set>

// Forward declarations
namespace llvm {
class BasicBlock;
} // namespace llvm

// Local libraries includes
#include "revng-c/RestructureCFGPass/BasicBlockNodeBB.h"
#include "revng-c/RestructureCFGPass/RegionCFGTree.h"
extern template class RegionCFG<llvm::BasicBlock *>;

using RegionCFGBB = RegionCFG<llvm::BasicBlock *>;

#endif // REVNGC_RESTRUCTURE_CFG_REGIONCFGTREEBB_H
