#ifndef REVNGC_RESTRUCTURE_CFG_FLATTENINGBB_H
#define REVNGC_RESTRUCTURE_CFG_FLATTENINGBB_H

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// Local libraries includes
#include "revng-c/RestructureCFGPass/Flattening.h"
extern template void flattenRegionCFGTree(RegionCFG<llvm::BasicBlock *> &Root);

#endif // REVNGC_RESTRUCTURE_CFG_FLATTENINGBB_H
