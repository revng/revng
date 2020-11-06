#ifndef REVNGC_RESTRUCTURE_CFG_BASICBLOCKNODEBB_H
#define REVNGC_RESTRUCTURE_CFG_BASICBLOCKNODEBB_H

//
// Copyright (c) rev.ng Srls 2017-2020.
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
