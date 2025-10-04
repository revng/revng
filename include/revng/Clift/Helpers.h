#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <optional>

#include "revng/Clift/Clift.h"
#include "revng/Pipeline/Location.h"
#include "revng/Pipes/Kinds.h"
#include "revng/Support/MetaAddress.h"

namespace mlir::clift {

inline MetaAddress getMetaAddress(clift::FunctionOp F) {
  if (auto L = pipeline::locationFromString(revng::ranks::Function,
                                            F.getHandle())) {
    auto [Key] = L->at(revng::ranks::Function);
    return Key;
  }
  return MetaAddress::invalid();
}

} // namespace mlir::clift
