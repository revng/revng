#pragma once

//
// Copyright rev.ng Labs Srl. See LICENSE.md for details.
//

#include "llvm/IR/Module.h"

#include "revng/Model/Binary.h"
#include "revng/Pipes/FunctionStringMap.h"

void decompile(llvm::Module &M,
               const model::Binary &Model,
               revng::pipes::FunctionStringMap &DecompiledFunctions);
