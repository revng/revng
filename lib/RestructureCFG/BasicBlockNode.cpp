/// \file BasicBlockNode.cpp
/// FunctionPass that applies the comb to the RegionCFG of a function

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/RestructureCFG/BasicBlockNodeImpl.h"

// Explicit instantiation for the `RegionCFG` template class.
template class BasicBlockNode<llvm::BasicBlock *>;
