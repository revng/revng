#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name Clift)
add_public_tablegen_target(MLIRCliftTransformsIncGen)
