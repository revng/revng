/// \file RegionCFGTree.cpp
/// FunctionPass that applies the comb to the RegionCFG of a function

//
// Copyright rev.ng Labs Srl. See LICENSE.md for details.
//

#include "revng-c/RestructureCFG/RegionCFGTreeImpl.h"

// Explicit instantiation for the `RegionCFG` template class.
template class RegionCFG<llvm::BasicBlock *>;

unsigned DuplicationCounter = 0;

unsigned UntangleTentativeCounter = 0;
unsigned UntanglePerformedCounter = 0;
