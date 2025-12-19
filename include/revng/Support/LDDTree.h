#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <map>
#include <string>

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

using LDDTree = std::map<std::string, llvm::SmallVector<std::string, 10>>;
void lddtree(LDDTree &Dependencies, llvm::StringRef Path, unsigned DepthLevel);
