#
# Copyright rev.ng Labs Srl. See LICENSE.md for details.
#

# This policy mimics the behavior set in `CMakePolicy.cmake` in the official
# LLVM repo. Can be dropped when it is dropped from there.
if(POLICY CMP0116)
  cmake_policy(SET CMP0116 OLD)
endif()

# This is required by the following `AddMLIR.cmake` file
include("${LLVM_DIR}/AddLLVM.cmake")

# This is needed for the `mlir_tablegen` and `add_public_tablegen_target`
# functions
include("${LLVM_DIR}/TableGen.cmake")

# This is needed for the `add_mlir_dialect` and `add_mlir_doc` functions
include("${MLIR_DIR}/AddMLIR.cmake")

add_mlir_dialect(CliftOps clift)
add_mlir_doc(CliftOps CliftDialect Dialects/ -gen-dialect-doc -dialect clift)

set(LLVM_TARGET_DEFINITIONS CliftTypes.td)
mlir_tablegen(CliftTypes.h.inc -gen-typedef-decls -typedefs-dialect=llvm)
mlir_tablegen(CliftTypes.cpp.inc -gen-typedef-defs -typedefs-dialect=llvm)
add_public_tablegen_target(MLIRCliftTypesIncGen)
add_dependencies(mlir-headers MLIRCliftTypesIncGen)

set(LLVM_TARGET_DEFINITIONS CliftAttributes.td)
mlir_tablegen(CliftAttributes.h.inc -gen-attrdef-decls)
mlir_tablegen(CliftAttributes.cpp.inc -gen-attrdef-defs)
add_public_tablegen_target(MLIRCliftAttributeIncGen)
add_dependencies(mlir-headers MLIRCliftAttributeIncGen)

set(LLVM_TARGET_DEFINITIONS CliftEnums.td)
mlir_tablegen(CliftEnums.h.inc -gen-enum-decls)
mlir_tablegen(CliftEnums.cpp.inc -gen-enum-defs)
add_public_tablegen_target(MLIRCliftEnumsIncGen)
add_dependencies(mlir-headers MLIRCliftEnumsIncGen)
