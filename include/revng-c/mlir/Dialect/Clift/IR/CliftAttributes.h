#pragma once

//
// Copyright (c) rev.ng Labs Srl. See LICENSE.md for details.
//

#include "mlir/IR/Attributes.h"

#include "revng-c/mlir/Dialect/Clift/IR/CliftEnums.h"

// This include should stay here for correct build procedure
#define GET_ATTRDEF_CLASSES
#include "revng-c/mlir/Dialect/Clift/IR/CliftAttributes.h.inc"
