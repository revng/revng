#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

namespace mlir::clift {

class FieldAttr;

} // namespace mlir::clift

#include "mlir/IR/Types.h"

#include "revng/Clift/CliftAttributes.h"

// Prevent reordering:
#include "revng/Clift/CliftTypeInterfacesBasic.h.inc"
// Prevent reordering:
#include "revng/Clift/CliftTypeInterfacesDefined.h.inc"
// Prevent reordering:
#include "revng/Clift/CliftTypeInterfacesClass.h.inc"
