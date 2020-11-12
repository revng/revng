/// \file BasicBlockNode.cpp
/// \brief FunctionPass that applies the comb to the RegionCFG of a function

//
// Copyright rev.ng Srls. See LICENSE.md for details.
//

#include "revng-c/RestructureCFGPass/BasicBlockNodeImpl.h"

// Explicit instantation for the `RegionCFG` template class.
template class BasicBlockNode<llvm::BasicBlock *>;
