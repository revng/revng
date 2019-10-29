/// \file Metaregion.cpp
/// \brief FunctionPass that applies the comb to the RegionCFG of a function

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// Local libraries includes
#include "revng-c/RestructureCFGPass/MetaRegionImpl.h"

// Explicit instantation for the `Metaregion` template class for
// llvm::BasicBlock type.
template class MetaRegion<llvm::BasicBlock *>;
