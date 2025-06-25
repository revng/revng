#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/Backend/DecompilePipe.h"
#include "revng/Pipes/StringMap.h"
#include "revng/Support/MetaAddress.h"

namespace ptml {
class ModelCBuilder;
}

namespace detail {
using DecompiledStringMap = revng::pipes::DecompileStringMap;
}

void printSingleCFile(ptml::ModelCBuilder &B,
                      const detail::DecompiledStringMap &Functions,
                      const std::set<MetaAddress> &Targets);
