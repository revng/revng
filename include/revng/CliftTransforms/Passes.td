//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#ifndef MLIR_CLIFT_PASSES
#define MLIR_CLIFT_PASSES

include "mlir/Pass/PassBase.td"

class Clift_Pass<string arg, string operation> : Pass<arg, operation> {
  let dependentDialects = ["clift::CliftDialect"];
}

def CliftCLegalization : Clift_Pass<"c-legalization", "clift::FunctionOp"> {
  let summary = "Legalize the input for emitting C code.";

  // Use the default target C implementation for clift-opt:
  let constructor = "mlir::clift::createCLegalizationPass(TargetCImplementation::Default)";
}

def CliftVerifyC : Clift_Pass<"verify-c", "mlir::ModuleOp"> {
  let summary = "Verify that the Clift semantics correspond to C semantics " #
                "in the specified target implementation.";
  let constructor = "mlir::clift::createVerifyCPass()";
}

def CliftEmitC : Clift_Pass<"emit-c", "mlir::ModuleOp"> {
  let summary = "Emit C code (optionally with PTML tags) from the Clift code.";
  let constructor = "mlir::clift::createEmitCPass()";

  let options = [
    Option<"EmitTags", "ptml", "bool", /*default=*/"false", "Emit PTML tags">,
    Option<"Output", "output", "std::string", /*default*/"\"-\"", "C output path">
  ];
}

#endif // MLIR_CLIFT_PASSES
