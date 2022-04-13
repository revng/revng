#pragma once

//
// Copyright rev.ng Labs Srl. See LICENSE.md for details.
//

#include "ContractedGraph.h"
#include "TypeFlowGraph.h"

namespace vma {

/// Contract the graph until you end with only two nodes
///
/// This implementation guarantees that NodesToColor and NodesToUncolor are
/// never merged, therefore they will be the only two nodes remaining at the
/// end. The algorithm is applied many times in a monte-carlo fashion.
/// \param G The graph to contract
/// \param BestCost The current cost, will be updated if a better one is found
/// \param BestNodesToColor Where to store the NodesToColor of the best solution
/// \param BestNodesToUncolor Where to store the NodesToUncolor of the best sol.
void karger(ContractedGraph &G,
            unsigned &BestCost,
            ContractedNode &BestNodesToColor,
            ContractedNode &BestNodesToUncolor);

/// Assign undecided nodes applying Karger one color at a time
///
/// This function reasons one color at a time. It creates a ContractedGraph for
/// each connected component of undecided nodes that have a given color among
/// their candidates. A probabilistic algorithm is then applied to find, in each
/// ContractedGraph, the minimal cut that divides the nodes of the current color
/// from nodes of other colors.
void minCut(TypeFlowGraph &TG);
} // namespace vma
