//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#ifndef MLIR_CLIFT_OP_TRAITS
#define MLIR_CLIFT_OP_TRAITS

include "mlir/IR/OpBase.td"

def Clift_NoFallthrough :
  NativeOpTrait<"clift::NoFallthrough">;

def Clift_ReturnsBoolean :
  NativeOpTrait<"clift::ReturnsBoolean">;

#endif
