/// \file RegionCFGTree.cpp
/// \brief FunctionPass that applies the comb to the RegionCFG of a function

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// Local libraries includes
#include "revng-c/RestructureCFGPass/RegionCFGTreeImpl.h"

// Explicit instantation for the `RegionCFG` template class.
template class RegionCFG<llvm::BasicBlock *>;

unsigned DuplicationCounter = 0;

unsigned UntangleTentativeCounter = 0;
unsigned UntanglePerformedCounter = 0;
