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
  let constructor = "clift::createTrivialReturnEliminationPass()";
}

def CliftLoopDetection : Clift_Pass<"loop-detection", "clift::FunctionOp"> {
  let summary = "Detect Clift loops";
  let constructor = "clift::createLoopDetectionPass()";
}

def CliftLabelMerging : Clift_Pass<"label-merging", "clift::FunctionOp"> {
  let summary = "Merge any two labels indicating the same statement.";
  let constructor = "clift::createLabelMergingPass()";
}

def CliftOptimizeStatements : Clift_Pass<"optimize-statements",
                                         "clift::FunctionOp"> {
  let summary = "Optimize Clift statements";
  let constructor = "clift::createOptimizeStatementsPass()";
  let options = RewritePassUtils.options;
}

def CliftOptimizeExpressions : Clift_Pass<"optimize-expressions",
                                          "clift::FunctionOp"> {
  let summary = "Optimize Clift expressions";
  let constructor = "clift::createOptimizeExpressionsPass()";
  let options = RewritePassUtils.options;
}

def CliftEmitFieldAccesses : Clift_Pass<"emit-field-accesses", "clift::FunctionOp"> {
  let summary = "Emit field accesses.";
  let constructor = "clift::createEmitFieldAccessesPass()";
}

def CliftTerminalBranchComplementHoisting
  : Clift_Pass<"terminal-branch-complement-hoisting", "clift::FunctionOp"> {

  let summary = "Hoist one side of a two-way branch operation, where at least "
                "one of the branches is non-fallthrough.";
  let constructor = "clift::createTerminalBranchComplementHoistingPass()";
}

def CliftTightenVariableScopes : Clift_Pass<"tighten-variable-scopes",
                                            "clift::FunctionOp"> {

  let summary = "Tighten variable scopes";
  let constructor = "clift::createVariableScopeTighteningPass()";
}

def CliftVariableInitializerHoisting : Clift_Pass<"hoist-variable-initializers",
                                                  "clift::FunctionOp"> {

  let summary = "Hoist variable initializers";
  let constructor = "clift::createVariableInitializerHoistingPass()";
}

def CliftPromoteBreakContinue : Clift_Pass<"promote-break-continue",
                                           "clift::FunctionOp"> {

  let summary = "Promote break_to/continue_to targeting the innermost enclosing "
                "loop to plain operand-less break/continue.";
  let constructor = "clift::createPromoteBreakContinuePass()";
}

def CliftCLegalization : Clift_Pass<"c-legalization", "clift::FunctionOp"> {
  let summary = "Legalize the input for emitting C code.";
  let constructor = "clift::createCLegalizationPass()";
}

def CliftImmediateRadixDeduction : Clift_Pass<"deduce-immediate-radices",
                                              "clift::FunctionOp"> {
  let summary = "Deduce most readable radices for integer immediates.";
  let constructor = "clift::createImmediateRadixDeductionPass()";
}

def CliftImplicitCastElision : Clift_Pass<"elide-implicit-casts",
                                          "clift::FunctionOp"> {
  let summary = "Mark implicit casts for elision by the backend.";
  let constructor = "clift::createImplicitCastElisionPass()";
}

def CliftVerifyC : Clift_Pass<"verify-c", "mlir::ModuleOp"> {
  let summary = "Verify that the Clift semantics correspond to C semantics " #
                "in the specified target implementation.";
  let constructor = "clift::createVerifyCPass()";
}

// A class gathering up options common for all the PTML emitters.
class Clift_EmitPTMLPass<string arg, string operation> : Clift_Pass<arg, operation> {
  let options = [
    Option<"EmitTags", "ptml", "bool", /* default = */"false", "Emit PTML tags">,
    Option<"Output", "output", "std::string", /* default = */"\"-\"", "C output path">,
    Option<"InlineStackFrameType",
           "inline-stack-frame",
           "bool",
           /* default = */"false",
           "Inline the body of each function's stack-frame struct type.">
  ];
}

def CliftEmitC : Clift_EmitPTMLPass<"emit-c", "mlir::ModuleOp"> {
  let summary = "Emit C code (optionally with PTML tags) from the Clift code.";
  let constructor = "clift::createEmitCPass()";
}

def CliftEmitTypeAndGlobalHeader : Clift_EmitPTMLPass<"emit-type-and-global-header",
                                                      "mlir::ModuleOp"> {
  // NOTE: this header is a subject to be split into a set of smaller ones.
  let summary = "Emit a C header containing the type system with some extras.";
  let constructor = "clift::createEmitTypeAndGlobalHeaderPass()";
}

def CliftEmitHelperHeader : Clift_EmitPTMLPass<"emit-helper-header",
                                               "mlir::ModuleOp"> {
  let summary = "Emit a C header containing *declarations* of all the helpers used in the module.";
  let constructor = "clift::createEmitHelperHeaderPass()";
}

// TODO: add the corresponding single-type pass if the need ever arises.

#endif // MLIR_CLIFT_PASSES
