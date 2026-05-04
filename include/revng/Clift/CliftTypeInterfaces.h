#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

namespace clift {

class CAttributeAttr;
class FieldAttr;

} // namespace clift

#include "mlir/IR/Types.h"

#include "revng/Clift/CliftAttributes.h"

// Prevent reordering:
#include "revng/Clift/CliftTypeInterfacesBasic.h.inc"
// Prevent reordering:
#include "revng/Clift/CliftTypeInterfacesObject.h.inc"
// Prevent reordering:
#include "revng/Clift/CliftTypeInterfacesValue.h.inc"
// Prevent reordering:
#include "revng/Clift/CliftTypeInterfacesDefined.h.inc"
// Prevent reordering:
#include "revng/Clift/CliftTypeInterfacesClass.h.inc"
// Prevent reordering:
#include "revng/Clift/CliftTypeInterfacesPrimitive.h.inc"
// Prevent reordering:
#include "revng/Clift/CliftTypeInterfacesIntegral.h.inc"
