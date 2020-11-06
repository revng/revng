#ifndef REVNGC_RESTRUCTURE_CFG_REGIONCFGTREEBB_H
#define REVNGC_RESTRUCTURE_CFG_REGIONCFGTREEBB_H

//
// Copyright (c) rev.ng Srls 2017-2020.
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
