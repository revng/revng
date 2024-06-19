#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/IR/Module.h"

#include "revng/EarlyFunctionAnalysis/ControlFlowGraphCache.h"
#include "revng/Model/Binary.h"
#include "revng/Pipes/StringMap.h"

#include "revng-c/Backend/DecompilePipe.h"

namespace detail {
using Container = revng::pipes::DecompileStringMap;
}

void decompile(ControlFlowGraphCache &Cache,
               llvm::Module &M,
               const model::Binary &Model,
               detail::Container &DecompiledFunctions,
               bool GeneratePlainC = false);
