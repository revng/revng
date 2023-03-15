#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "Helpers.h"

/// Ranks the nodes using specified ranking strategy.
template<RankingStrategy Strategy>
RankContainer rankNodes(InternalGraph &Graph);

/// Ranks the nodes using specified ranking strategy.
template<RankingStrategy Strategy>
RankContainer rankNodes(InternalGraph &Graph, int64_t DiamondBound);

/// Updates node ranking after the graph was modified.
///
/// Guarantees ranking consistency i.e. that each node has
/// a rank greater than its predecessors.
///
/// Be careful, rank order is NOT preserved.
/// \note: There used to be an additional parameter for that, but it was never
/// actually implemented, so I removed it.
RankContainer &updateRanks(InternalGraph &Graph, RankContainer &Ranks);
