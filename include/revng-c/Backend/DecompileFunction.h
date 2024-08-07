#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/IR/Module.h"

#include "revng/EarlyFunctionAnalysis/ControlFlowGraphCache.h"
#include "revng/Model/Binary.h"
#include "revng/Pipes/StringMap.h"

#include "revng-c/Backend/DecompilePipe.h"

namespace ptml {
class CTypeBuilder;
}

std::string decompile(ControlFlowGraphCache &Cache,
                      llvm::Function &F,
                      const model::Binary &Model,
                      ptml::CTypeBuilder &B);
