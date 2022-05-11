/// \file Graph.cpp
/// \brief

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/Support/FormatVariadic.h"

#include "revng/Model/Binary.h"
#include "revng/Yield/Internal/Function.h"
#include "revng/Yield/SVG.h"

std::string yield::svg::controlFlow(const yield::Function &Function,
                                    const model::Binary &Binary) {
  return Function.Name;
}
