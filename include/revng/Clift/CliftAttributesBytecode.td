//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#ifndef MLIR_CLIFT_ATTRIBUTE_BYTECODE
#define MLIR_CLIFT_ATTRIBUTE_BYTECODE

include "mlir/IR/AttrTypeBase.td"

include "revng/Clift/CliftDialect.td"

// This is a helper attribute used for emitting the bytecode representation of
// (potentially recursive) class types.
def Clift_BytecodeClassAttr : AttrDef<Clift_Dialect, "BytecodeClass"> {
  let parameters = (ins "mlir::clift::ClassType":$type);
}

#endif
