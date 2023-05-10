#pragma once

/// \file ABISummary.h

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <map>
#include <set>

#include "revng/Support/BasicBlockID.h"

namespace efa {

using RegisterSet = std::set<const llvm::GlobalVariable *>;

struct ABISummary {
  RegisterSet Arguments;
  RegisterSet Returns;
};

} // namespace efa
