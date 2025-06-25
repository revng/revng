#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/IR/Module.h"

#include "revng/Backend/DecompilePipe.h"
#include "revng/EarlyFunctionAnalysis/ControlFlowGraphCache.h"
#include "revng/Model/Binary.h"
#include "revng/Pipes/StringMap.h"

namespace ptml {
class ModelCBuilder;
}

std::string decompile(ControlFlowGraphCache &Cache,
                      llvm::Function &F,
                      const model::Binary &Model,
                      ptml::ModelCBuilder &B);
