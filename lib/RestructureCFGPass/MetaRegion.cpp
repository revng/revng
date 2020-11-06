/// \file Metaregion.cpp
/// \brief FunctionPass that applies the comb to the RegionCFG of a function

//
// Copyright (c) rev.ng Srls 2017-2020.
//

// Local libraries includes
#include "revng-c/RestructureCFGPass/MetaRegionImpl.h"

// Explicit instantation for the `Metaregion` template class for
// llvm::BasicBlock type.
template class MetaRegion<llvm::BasicBlock *>;
