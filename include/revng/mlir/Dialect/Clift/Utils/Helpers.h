#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <optional>

#include "mlir/IR/FunctionInterfaces.h"

#include "revng/Support/IRHelpers.h"
#include "revng/Support/MetaAddress.h"

namespace mlir::clift {

inline MetaAddress getMetaAddress(mlir::FunctionOpInterface F) {
  if (auto Attr = F->getAttrOfType<mlir::StringAttr>(FunctionEntryMDName))
    return MetaAddress::fromString(Attr.getValue());
  return MetaAddress::invalid();
}

} // namespace mlir::clift
