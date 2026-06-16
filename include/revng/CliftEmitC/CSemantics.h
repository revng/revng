#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/Clift/Clift.h"

// TODO: does this really belong with the emitters?

mlir::LogicalResult verifyCSemantics(mlir::ModuleOp Module);
