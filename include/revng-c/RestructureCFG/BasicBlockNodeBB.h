#pragma once

//
// Copyright rev.ng Labs Srl. See LICENSE.md for details.
//

// Forward declarations
namespace llvm {
class BasicBlock;
} // namespace llvm

#include "revng-c/RestructureCFG/BasicBlockNode.h"
extern template class BasicBlockNode<llvm::BasicBlock *>;

using BasicBlockNodeBB = BasicBlockNode<llvm::BasicBlock *>;
