//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#ifndef MLIR_CLIFT_PASSES
#define MLIR_CLIFT_PASSES

include "mlir/Pass/PassBase.td"
include "mlir/Rewrite/PassUtil.td"

class Clift_Pass<string arg, string operation> : Pass<arg, operation> {
  let dependentDialects = ["clift::CliftDialect"];
}

def CliftTrivialReturnElimination : Clift_Pass<"trivial-return-elimination",
                                               "clift::FunctionOp"> {
  let summary = "Remove any void return statements where control falls through "
                "to the end of the function.";
  let constructor = "mlir::clift::createTrivialReturnEliminationPass()";
}

def CliftLoopDetection : Clift_Pass<"loop-detection", "clift::FunctionOp"> {
  let summary = "Detect Clift loops";
  let constructor = "mlir::clift::createLoopDetectionPass()";
}

def CliftLabelMerging : Clift_Pass<"label-merging", "clift::FunctionOp"> {
  let summary = "Merge any two labels indicating the same statement.";
  let constructor = "mlir::clift::createLabelMergingPass()";
}

def CliftOptimizeStatements : Clift_Pass<"optimize-statements",
                                         "clift::FunctionOp"> {
  let summary = "Optimize Clift statements";
  let constructor = "mlir::clift::createOptimizeStatementsPass()";
  let options = RewritePassUtils.options;
}

def CliftOptimizeExpressions : Clift_Pass<"optimize-expressions",
                                          "clift::FunctionOp"> {
  let summary = "Optimize Clift expressions";
  let constructor = "mlir::clift::createOptimizeExpressionsPass()";
  let options = RewritePassUtils.options;
}

def CliftEmitFieldAccesses : Clift_Pass<"emit-field-accesses", "clift::FunctionOp"> {
  let summary = "Emit field accesses.";
  let constructor = "mlir::clift::createEmitFieldAccessesPass()";
}

def CliftTerminalBranchComplementHoisting
  : Clift_Pass<"terminal-branch-complement-hoisting", "clift::FunctionOp"> {

  let summary = "Hoist one side of a two-way branch operation, where at least "
                "one of the branches is non-fallthrough.";
  let constructor = "mlir::clift::createTerminalBranchComplementHoistingPass()";
}

def CliftCLegalization : Clift_Pass<"c-legalization", "clift::FunctionOp"> {
  let summary = "Legalize the input for emitting C code.";

  // Use the default target C implementation for clift-opt:
  let constructor = "mlir::clift::createCLegalizationPass(TargetCImplementation::Default)";
}

def CliftImmediateRadixDeduction : Clift_Pass<"deduce-immediate-radices",
                                              "clift::FunctionOp"> {
  let summary = "Deduce most readable radices for integer immediates.";
  let constructor = "mlir::clift::createImmediateRadixDeductionPass()";
}

def CliftVerifyC : Clift_Pass<"verify-c", "mlir::ModuleOp"> {
  let summary = "Verify that the Clift semantics correspond to C semantics " #
                "in the specified target implementation.";
  let constructor = "mlir::clift::createVerifyCPass()";
}

// A class gathering up options common for all the PTML emitters.
class Clift_EmitPTMLPass<string arg, string operation> : Clift_Pass<arg, operation> {
  let options = [
    Option<"EmitTags", "ptml", "bool", /*default=*/"false", "Emit PTML tags">,
    Option<"Output", "output", "std::string", /*default*/"\"-\"", "C output path">
  ];
}

def CliftEmitC : Clift_EmitPTMLPass<"emit-c", "mlir::ModuleOp"> {
  let summary = "Emit C code (optionally with PTML tags) from the Clift code.";
  let constructor = "mlir::clift::createEmitCPass()";
}

#endif // MLIR_CLIFT_PASSES
