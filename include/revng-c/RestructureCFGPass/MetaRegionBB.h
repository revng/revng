#ifndef REVNGC_RESTRUCTURE_CFG_METAREGIONBB_H
#define REVNGC_RESTRUCTURE_CFG_METAREGIONBB_H

//
// Copyright rev.ng Srls. See LICENSE.md for details.
//

// Forward declarations
namespace llvm {
class BasicBlock;
} // namespace llvm

#include "revng-c/RestructureCFGPass/MetaRegion.h"
extern template class MetaRegion<llvm::BasicBlock *>;

using MetaRegionBB = MetaRegion<llvm::BasicBlock *>;

#endif // REVNGC_RESTRUCTURE_CFG_METAREGIONBB_H
