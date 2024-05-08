#pragma once

//
// This file is distributed under the MIT License. See LICENSE.mit for details.
//

#include <map>
#include <string>

#include "llvm/ADT/SmallVector.h"

using LDDTree = std::map<std::string, llvm::SmallVector<std::string, 10>>;
void lddtree(LDDTree &Dependencies,
             const std::string &Path,
             unsigned DepthLevel);
