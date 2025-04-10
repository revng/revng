#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

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

set(LLVM_TARGET_DEFINITIONS CliftTypeInterfacesBasic.td)
mlir_tablegen(CliftTypeInterfacesBasic.h.inc -gen-type-interface-decls)
mlir_tablegen(CliftTypeInterfacesBasic.cpp.inc -gen-type-interface-defs)
set(LLVM_TARGET_DEFINITIONS CliftTypeInterfacesDefined.td)
mlir_tablegen(CliftTypeInterfacesDefined.h.inc -gen-type-interface-decls)
mlir_tablegen(CliftTypeInterfacesDefined.cpp.inc -gen-type-interface-defs)
set(LLVM_TARGET_DEFINITIONS CliftTypeInterfacesClass.td)
mlir_tablegen(CliftTypeInterfacesClass.h.inc -gen-type-interface-decls)
mlir_tablegen(CliftTypeInterfacesClass.cpp.inc -gen-type-interface-defs)
add_public_tablegen_target(MLIRCliftTypeInterfaceIncGen)
add_dependencies(mlir-headers MLIRCliftTypeInterfaceIncGen)

set(LLVM_TARGET_DEFINITIONS CliftAttrInterfaces.td)
mlir_tablegen(CliftAttrInterfaces.h.inc -gen-attr-interface-decls)
mlir_tablegen(CliftAttrInterfaces.cpp.inc -gen-attr-interface-defs)
add_public_tablegen_target(MLIRCliftAttrInterfaceIncGen)
add_dependencies(mlir-headers MLIRCliftAttrInterfaceIncGen)

set(LLVM_TARGET_DEFINITIONS CliftOpInterfacesBasic.td)
mlir_tablegen(CliftOpInterfacesBasic.h.inc -gen-op-interface-decls)
mlir_tablegen(CliftOpInterfacesBasic.cpp.inc -gen-op-interface-defs)
set(LLVM_TARGET_DEFINITIONS CliftOpInterfacesStatement.td)
mlir_tablegen(CliftOpInterfacesStatement.h.inc -gen-op-interface-decls)
mlir_tablegen(CliftOpInterfacesStatement.cpp.inc -gen-op-interface-defs)
add_public_tablegen_target(MLIRCliftOpInterfaceIncGen)
add_dependencies(mlir-headers MLIRCliftOpInterfaceIncGen)
