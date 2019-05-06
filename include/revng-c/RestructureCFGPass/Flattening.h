#ifndef REVNGC_RESTRUCTURE_CFG_FLATTENING_H
#define REVNGC_RESTRUCTURE_CFG_FLATTENING_H

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// forward declarations
template<class NodeT>
class RegionCFG;

template<class NodeT>
void flattenRegionCFGTree(RegionCFG<NodeT> &Root);

#endif // REVNGC_RESTRUCTURE_CFG_FLATTENING_H
