#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/ADT/SetVector.h"
#include "llvm/IR/BasicBlock.h"

/// Helper function to retrieve the successors of a block on the `ScopeGraph`
llvm::SmallSetVector<llvm::BasicBlock *, 2>
getScopeGraphSuccessors(llvm::BasicBlock *N);

/// Helper function to retrieve the predecessors of a block on the `ScopeGraph`
llvm::SmallSetVector<llvm::BasicBlock *, 2>
getScopeGraphPredecessors(llvm::BasicBlock *N);

/// Helper function to collect all the nodes, on the `ScopeGraph`, between a
/// block and its immediate postdominator, using a _dfs_ext`
llvm::SmallVector<llvm::BasicBlock *>
getNodesInScope(llvm::BasicBlock *ScopeEntryBlock,
                llvm::BasicBlock *PostDominator);
