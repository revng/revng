#pragma once

//
// Copyright rev.ng Srls. See LICENSE.md for details.
//

#include "revng-c/RestructureCFGPass/ASTTree.h"

extern unsigned ShortCircuitCounter;
extern unsigned TrivialShortCircuitCounter;

// Forward declarations
namespace MarkForSerialization {
class Analysis;
}

void beautifyAST(llvm::Function &F,
                 ASTTree &CombedAST,
                 MarkForSerialization::Analysis &Mark);
