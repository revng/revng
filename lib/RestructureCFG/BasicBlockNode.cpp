/// \file BasicBlockNode.cpp
/// FunctionPass that applies the comb to the RegionCFG of a function

//
// Copyright rev.ng Labs Srl. See LICENSE.md for details.
//

#include "revng-c/RestructureCFG/BasicBlockNodeImpl.h"

// Explicit instantiation for the `RegionCFG` template class.
template class BasicBlockNode<llvm::BasicBlock *>;
