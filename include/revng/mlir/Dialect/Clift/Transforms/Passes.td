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

def CliftLoopDetection : Clift_Pass<"loop-detection", "clift::FunctionOp"> {
  let summary = "Detect Clift loops";
  let constructor = "mlir::clift::createLoopDetectionPass()";
}

def CliftBranchEqualization : Clift_Pass<"branch-equalization",
                                         "clift::FunctionOp"> {
  let summary = "Equalize Clift branches";
  let constructor = "mlir::clift::createBranchEqualizationPass()";
}

def CliftBeautifyStatements : Clift_Pass<"beautify-statements",
                                         "clift::FunctionOp"> {
  let summary = "Beautify Clift statements";
  let constructor = "mlir::clift::createBeautifyStatementsPass()";
  let options = RewritePassUtils.options;
}

def CliftReturnIntoGotoConversion : Clift_Pass<"return-into-goto",
                                               "clift::FunctionOp"> {
  let summary = "Convert return statements into gotos";
  let constructor = "mlir::clift::createReturnIntoGotoConversionPass()";
}

def CliftGotoIntoReturnConversion : Clift_Pass<"goto-into-return",
                                               "clift::FunctionOp"> {
  let summary = "Convert gotos into return statements";
  let constructor = "mlir::clift::createGotoIntoReturnConversionPass()";
}

def CliftBeautifyExpressions : Clift_Pass<"beautify-expressions",
                                          "clift::FunctionOp"> {
  let summary = "Beautify Clift expressions";
  let constructor = "mlir::clift::createBeautifyExpressionsPass()";
  let options = RewritePassUtils.options;
}

def CliftExpressionRootSimplification : Clift_Pass<"simplify-expression-roots",
                                                   "clift::FunctionOp"> {
  let summary = "Simplify Clift expression roots";
  let constructor = "mlir::clift::createExpressionRootSimplificationPass()";
}

def CliftImmediateRadixDeduction : Clift_Pass<"deduce-immediate-radices",
                                              "clift::FunctionOp"> {
  let summary = "Deduce most readable radices for integer immediates";
  let constructor = "mlir::clift::createImmediateRadixDeductionPass()";
}

def CliftVerifyC : Clift_Pass<"verify-c", "mlir::ModuleOp"> {
  let summary = "Verify that the Clift semantics correspond to C semantics " #
                "in the specified target implementation.";
  let constructor = "mlir::clift::createVerifyCPass()";
}

#endif // MLIR_CLIFT_PASSES
