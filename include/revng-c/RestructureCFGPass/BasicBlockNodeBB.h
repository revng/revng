#ifndef REVNGC_RESTRUCTURE_CFG_BASICBLOCKNODEBB_H
#define REVNGC_RESTRUCTURE_CFG_BASICBLOCKNODEBB_H

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// Forward declarations
namespace llvm {
class BasicBlock;
} // namespace llvm

// Local libraries includes
#include "revng-c/RestructureCFGPass/BasicBlockNode.h"
extern template class BasicBlockNode<llvm::BasicBlock *>;

using BasicBlockNodeBB = BasicBlockNode<llvm::BasicBlock *>;

#endif // REVNGC_RESTRUCTURE_CFG_BASICBLOCKNODEBB_H
