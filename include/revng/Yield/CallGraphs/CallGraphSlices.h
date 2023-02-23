#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/Yield/Graph.h"

class MetaAddress;

namespace yield::calls {

/// Produces a forwards facing slice of the graph starting from a single node.
///
/// Such a slice guarantees that:
/// - Only nodes forward-reachable from the `SlicePoint` are present.
/// - A node can have at most one predecessor, fake "reference" nodes without
///   trees are added in place of some real node to guarantee that, for example:
///
///   ```
///     A -> B             ||               ||
///     A -> C             ||   A - C - D   ||
///     B -> C             ||    \ /        ||
///     C -> D             ||     B         ||
///   ```
///   becomes
///   ```
///     A -> B             ||               ||
///     A -> C             ||   A - C - D   ||
///     B -> C'            ||    \          ||
///     C -> D             ||     B - C'    ||
///   ```
///   where `C'` is a fake node added to reference the `C` subtree without
///   being connected to it.
/// - There are no backwards facing edges (the targets of those are also
///   replaced by fake "reference" nodes).
///
/// \note: this makes a copy of the graph, as such `Input` is not affected.
Graph makeCalleeTree(const Graph &Input,
                     const BasicBlockID &SlicePoint = BasicBlockID());

/// Produces a backwards facing slice of the graph starting from a single node.
///
/// It is exactly the same as \see makeCalleeTree except it works in
/// the opposite direction (it makes sure all the predecessors are preserved).
Graph makeCallerTree(const Graph &Input,
                     const BasicBlockID &SlicePoint = BasicBlockID());

} // namespace yield::calls
