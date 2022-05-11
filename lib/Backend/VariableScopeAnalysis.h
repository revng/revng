#pragma once

//
// Copyright rev.ng Labs Srl. See LICENSE.md for details.
//

#include "llvm/ADT/SmallPtrSet.h"

namespace llvm {

class Function;
class Value;

} // end namespace llvm

class ASTTree;

llvm::SmallPtrSet<const llvm::Value *, 32>
collectLocalVariables(const llvm::Function &F);

bool hasLoopDispatchers(const ASTTree &GHAST);
