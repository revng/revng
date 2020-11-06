/// \file BasicBlockNode.cpp
/// \brief FunctionPass that applies the comb to the RegionCFG of a function

//
// Copyright (c) rev.ng Srls 2017-2020.
//

// Local libraries includes
#include "revng-c/RestructureCFGPass/BasicBlockNodeImpl.h"

// Explicit instantation for the `RegionCFG` template class.
template class BasicBlockNode<llvm::BasicBlock *>;
