/// \file Restructure.cpp
/// \brief FunctionPass that applies the comb to the RegionCFG of a function

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// Local libraries includes
#include "revng-c/RestructureCFGPass/BasicBlockNodeImpl.h"

using namespace llvm;

// Explicit instantation for the `RegionCFG` template class.
template class BasicBlockNode<llvm::BasicBlock *>;
