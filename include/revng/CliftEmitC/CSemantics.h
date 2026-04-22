#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/Clift/Clift.h"
#include "revng/Support/CTarget.h"

// TODO: does this really belong with the emitters?

mlir::LogicalResult verifyCSemantics(mlir::ModuleOp Module,
                                     const TargetCImplementation &Target);
