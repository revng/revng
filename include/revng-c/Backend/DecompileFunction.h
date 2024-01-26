#pragma once

//
// Copyright rev.ng Labs Srl. See LICENSE.md for details.
//

#include "llvm/IR/Module.h"

#include "revng/EarlyFunctionAnalysis/FunctionMetadataCache.h"
#include "revng/Model/Binary.h"
#include "revng/Pipes/StringMap.h"

#include "revng-c/Backend/CDecompilationPipe.h"

namespace detail {
using Container = revng::pipes::DecompiledCCodeInYAMLStringMap;
}

void decompile(FunctionMetadataCache &Cache,
               llvm::Module &M,
               const model::Binary &Model,
               detail::Container &DecompiledFunctions);
