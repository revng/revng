//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#ifndef MLIR_CLIFT_OP_INTERFACES
#define MLIR_CLIFT_OP_INTERFACES

// The MLIR tablegen tool emits declarations in alphabetical order, causing
// incorrect emission of interface hierarchies. The op interface declarations
// are split up into separate files in order to emit in the correct order.

include "revng/mlir/Dialect/Clift/IR/CliftOpInterfacesStatement.td"

#endif
