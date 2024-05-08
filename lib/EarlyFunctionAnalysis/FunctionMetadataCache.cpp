//
// This file is distributed under the MIT License. See LICENSE.mit for details.
//

#include "revng/EarlyFunctionAnalysis/FunctionMetadataCache.h"

char FunctionMetadataCachePass::ID = '_';

llvm::AnalysisKey FunctionMetadataCacheAnalysis::Key;

static llvm::RegisterPass<FunctionMetadataCachePass> _("metadata-cache",
                                                       "Create metadata cache "
                                                       "to be "
                                                       "used by later passes",
                                                       true,
                                                       true);
