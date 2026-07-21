#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/BasicAnalyses/GeneratedCodeBasicInfo.h"
#include "revng/EarlyFunctionAnalysis/ControlFlowGraphCache.h"
#include "revng/Model/Binary.h"

void collectFunctionsFromUnusedAddresses(llvm::Module &M,
                                         GeneratedCodeBasicInfo &GCBI,
                                         model::Binary &Binary,
                                         ControlFlowGraphCache &FMC);
