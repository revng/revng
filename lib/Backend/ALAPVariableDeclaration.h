#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <unordered_map>

#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"

class ASTTree;
class ASTNode;
namespace llvm {
class CallInst;
}

using LocalVarDeclSet = llvm::SmallSetVector<const llvm::CallInst *, 4>;
using ASTVarDeclMap = std::unordered_map<const ASTNode *, LocalVarDeclSet>;

using PendingVariableListType = llvm::SmallVector<const llvm::CallInst *>;

extern ASTVarDeclMap
computeVarDeclMap(const ASTTree &GHAST,
                  PendingVariableListType &PendingVariables);
