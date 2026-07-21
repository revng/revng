#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/Pass.h"

#include "revng/BasicAnalyses/GeneratedCodeBasicInfo.h"
#include "revng/Model/Binary.h"
#include "revng/Support/IRHelpers.h"
#include "revng/Support/MetaAddress.h"

void collectFunctionsFromCallees(llvm::Module &M,
                                 GeneratedCodeBasicInfo &GCBI,
                                 model::Binary &Binary);
