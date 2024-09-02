#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/Support/raw_ostream.h"

#include "revng/Pipes/StringMap.h"
#include "revng/Support/MetaAddress.h"

#include "revng-c/Backend/DecompilePipe.h"
#include "revng-c/Support/PTMLC.h"

namespace detail {
using DecompiledStringMap = revng::pipes::DecompileStringMap;
}

void printSingleCFile(llvm::raw_ostream &Out,
                      ptml::CBuilder &B,
                      const detail::DecompiledStringMap &Functions,
                      const std::set<MetaAddress> &Targets);
