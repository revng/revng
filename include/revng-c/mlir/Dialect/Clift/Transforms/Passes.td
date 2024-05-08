//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#ifndef MLIR_CLIFT_PASSES
#define MLIR_CLIFT_PASSES

include "mlir/Pass/PassBase.td"

def CliftImportModelTypes : Pass<"import-model-types", "mlir::ModuleOp"> {
  let summary = "Import model types and add references to them in clift.undef ops";
  let constructor = "mlir::clift::createImportModelTypesPass()";
  let dependentDialects = ["clift::CliftDialect"];

  let options = [
    Option<"ImportAll", "import-all", "bool", /*default=*/"false", "Import all model types">,
    Option<"Model", "model", "mlir::clift::ModelOptionType", /*default=*/"nullptr", "Model path">
  ];
}

#endif // MLIR_CLIFT_PASSES
