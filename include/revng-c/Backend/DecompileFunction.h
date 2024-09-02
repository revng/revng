#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/IR/Module.h"

#include "revng/EarlyFunctionAnalysis/ControlFlowGraphCache.h"
#include "revng/Model/Binary.h"
#include "revng/Pipes/StringMap.h"

#include "revng-c/Backend/DecompilePipe.h"

using TypeDefinitionSet = std::set<const model::TypeDefinition *>;
using InlineableTypesMap = std::unordered_map<const model::Function *,
                                              TypeDefinitionSet>;

std::string decompile(ControlFlowGraphCache &Cache,
                      llvm::Function &F,
                      const model::Binary &Model,
                      const InlineableTypesMap &StackTypes,
                      bool GeneratePlainC);
