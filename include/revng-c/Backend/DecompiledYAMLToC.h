#pragma once

//
// Copyright rev.ng Labs Srl. See LICENSE.md for details.
//

#include "llvm/Support/raw_ostream.h"

#include "revng/Pipes/FunctionStringMap.h"
#include "revng/Support/MetaAddress.h"

#include "revng-c/Backend/CDecompilationPipe.h"
#include "revng-c/Support/PTMLC.h"

namespace detail {
using DecompiledStringMap = revng::pipes::DecompiledCCodeInYAMLStringMap;
}

void printSingleCFile(llvm::raw_ostream &Out,
                      ptml::PTMLCBuilder &B,
                      const detail::DecompiledStringMap &Functions,
                      const std::set<MetaAddress> &Targets);
