/// \file Flattening.h
/// \brief Helper functions for flattening the RegionCFGTree after combing

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//
#ifndef REVNGC_RESTRUCTURE_CFG_FLATTENING_H
#define REVNGC_RESTRUCTURE_CFG_FLATTENING_H

// forward declarations
class RegionCFG;

void flattenRegionCFGTree(RegionCFG &Root);

#endif // REVNGC_RESTRUCTURE_CFG_FLATTENING_H
