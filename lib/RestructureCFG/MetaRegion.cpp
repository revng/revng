/// \file Metaregion.cpp
/// FunctionPass that applies the comb to the RegionCFG of a function

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng-c/RestructureCFG/MetaRegionImpl.h"

// Explicit instantiation for the `Metaregion` template class for
// llvm::BasicBlock type.
template class MetaRegion<llvm::BasicBlock *>;
