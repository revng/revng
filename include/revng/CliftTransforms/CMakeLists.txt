#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name Clift)

set(LLVM_TARGET_DEFINITIONS BooleanNegations.td)
mlir_tablegen(BooleanNegations.h.inc -gen-rewriters)

set(LLVM_TARGET_DEFINITIONS CastCanonicalization.td)
mlir_tablegen(CastCanonicalization.h.inc -gen-rewriters)

set(LLVM_TARGET_DEFINITIONS Expressions.td)
mlir_tablegen(Expressions.h.inc -gen-rewriters)

set(LLVM_TARGET_DEFINITIONS ExpressionPostProcessing.td)
mlir_tablegen(ExpressionPostProcessing.h.inc -gen-rewriters)

set(LLVM_TARGET_DEFINITIONS BooleanCastElimination.td)
mlir_tablegen(BooleanCastElimination.h.inc -gen-rewriters)

add_public_tablegen_target(MLIRCliftTransformsIncGen)
