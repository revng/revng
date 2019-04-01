/// \file Flattening.h
/// \brief Helper functions for flattening the RegionCFGTree after combing

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//
#ifndef REVNGC_RESTRUCTURE_CFG_FLATTENING_H
#define REVNGC_RESTRUCTURE_CFG_FLATTENING_H

// LLVM includes
#include "llvm/IR/BasicBlock.h"

// forward declarations
class RegionCFG;

// BBNodeToBBMap is a map that contains the original link to the LLVM basic
// block.
using BBNodeToBBMap = std::map<BasicBlockNode *, llvm::BasicBlock *>;

void flattenRegionCFGTree(RegionCFG &Root, BBNodeToBBMap &OriginalBB);

#endif // REVNGC_RESTRUCTURE_CFG_FLATTENING_H
