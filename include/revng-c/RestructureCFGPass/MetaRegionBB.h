#ifndef REVNGC_RESTRUCTURE_CFG_METAREGIONBB_H
#define REVNGC_RESTRUCTURE_CFG_METAREGIONBB_H

//
// Copyright (c) rev.ng Srls 2017-2020.
//

// Forward declarations
namespace llvm {
class BasicBlock;
} // namespace llvm

// Local libraries include
#include "revng-c/RestructureCFGPass/MetaRegion.h"
extern template class MetaRegion<llvm::BasicBlock *>;

using MetaRegionBB = MetaRegion<llvm::BasicBlock *>;

#endif // REVNGC_RESTRUCTURE_CFG_METAREGIONBB_H
