#pragma once

//
// Copyright rev.ng Srls. See LICENSE.md for details.
//

#include "revng-c/MarkForSerialization/MarkForSerializationFlags.h"
#include "revng-c/RestructureCFGPass/ASTTree.h"

extern void beautifyAST(llvm::Function &F,
                        ASTTree &CombedAST,
                        const SerializationMap &Mark);
