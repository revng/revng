//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#ifndef MLIR_CLIFT_OP_TRAITS
#define MLIR_CLIFT_OP_TRAITS

include "mlir/IR/OpBase.td"

class Clift_NativeOpTrait<string name, list<Trait> traits = []>
  : NativeOpTrait<name, traits> {

  let cppNamespace = "clift";
}

def Clift_NoFallthrough : Clift_NativeOpTrait<"NoFallthrough">;
def Clift_AssignsLoopLabels : Clift_NativeOpTrait<"AssignsLoopLabels">;
def Clift_ReturnsBoolean : Clift_NativeOpTrait<"ReturnsBoolean">;

#endif
