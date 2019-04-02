#ifndef REVNGC_CDECOMPILERBEAUTIFY_H
#define REVNGC_CDECOMPILERBEAUTIFY_H

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// local libraries includes
#include "revng-c/RestructureCFGPass/ASTTree.h"

// Forward declarations
namespace MarkForSerialization {
class Analysis;
}

void beautifyAST(llvm::Function &F,
                 ASTTree &CombedAST,
                 MarkForSerialization::Analysis &Mark);

#endif // REVNGC_CDECOMPILERBEAUTIFY_H
