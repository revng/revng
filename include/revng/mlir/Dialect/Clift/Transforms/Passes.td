//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#ifndef MLIR_CLIFT_PASSES
#define MLIR_CLIFT_PASSES

include "mlir/Pass/PassBase.td"

def CliftVerifyC : Pass<"verify-c", "mlir::ModuleOp"> {
  let summary = "Verify that the Clift semantics correspond to C semantics " #
                "in the specified target implementation.";
  let constructor = "mlir::clift::createVerifyCPass()";
  let dependentDialects = ["clift::CliftDialect"];
}

#endif // MLIR_CLIFT_PASSES
