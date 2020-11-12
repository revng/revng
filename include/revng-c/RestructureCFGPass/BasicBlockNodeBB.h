#pragma once

//
// Copyright rev.ng Srls. See LICENSE.md for details.
//

// Forward declarations
namespace llvm {
class BasicBlock;
} // namespace llvm

#include "revng-c/RestructureCFGPass/BasicBlockNode.h"
extern template class BasicBlockNode<llvm::BasicBlock *>;

using BasicBlockNodeBB = BasicBlockNode<llvm::BasicBlock *>;
