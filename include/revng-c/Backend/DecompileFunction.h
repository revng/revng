#pragma once

//
// Copyright rev.ng Labs Srl. See LICENSE.md for details.
//

#include "revng/Model/Binary.h"

#include "revng-c/Backend/VariableScopeAnalysisPass.h"
#include "revng-c/InitModelTypes/InitModelTypes.h"
#include "revng-c/MarkForSerialization/MarkAnalysis.h"
#include "revng-c/RestructureCFGPass/ASTTree.h"

using ValueSet = VariableScopeAnalysisPass::ValuePtrSet;

void decompileFunction(const llvm::Function &F,
                       const ASTTree &CombedAST,
                       const model::Binary &Model,
                       llvm::raw_ostream &O,
                       const ValueSet &TopScopeVariables,
                       bool NeedsLocalStateVar);
