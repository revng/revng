/// \file Flattening.h
/// \brief Helper functions for flattening the RegionCFGTree after combing

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//
#ifndef REVNGC_RESTRUCTURE_CFG_FLATTENING_H
#define REVNGC_RESTRUCTURE_CFG_FLATTENING_H

// std includes
#include <vector>

// local includes
#include "revng-c/RestructureCFGPass/RegionCFGTree.h"

void flattenRegionCFGTree(RegionCFG &Root, std::vector<RegionCFG> &Regions);

#endif // REVNGC_RESTRUCTURE_CFG_FLATTENING_H
