//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/EarlyFunctionAnalysis/ControlFlowGraphCache.h"

char ControlFlowGraphCachePass::ID = '_';

llvm::AnalysisKey ControlFlowGraphCacheAnalysis::Key;

static llvm::RegisterPass<ControlFlowGraphCachePass> _("metadata-cache",
                                                       "Create metadata cache "
                                                       "to be "
                                                       "used by later passes",
                                                       true,
                                                       true);
