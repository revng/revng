#ifndef REVNGC_CDECOMPILERBEAUTIFY_H
#define REVNGC_CDECOMPILERBEAUTIFY_H

//
// Copyright (c) rev.ng Srls 2017-2020.
//

// local libraries includes
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

#endif // REVNGC_CDECOMPILERBEAUTIFY_H
