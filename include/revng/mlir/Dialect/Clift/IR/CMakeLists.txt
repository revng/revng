#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

set(LLVM_TARGET_DEFINITIONS CliftAttributes.td)
mlir_tablegen(CliftAttributes.h.inc -gen-attrdef-decls)
mlir_tablegen(CliftAttributes.cpp.inc -gen-attrdef-defs)

set(LLVM_TARGET_DEFINITIONS CliftEnums.td)
mlir_tablegen(CliftEnums.h.inc -gen-enum-decls)
mlir_tablegen(CliftEnums.cpp.inc -gen-enum-defs)

set(LLVM_TARGET_DEFINITIONS CliftTypeInterfacesBasic.td)
mlir_tablegen(CliftTypeInterfacesBasic.h.inc -gen-type-interface-decls)
mlir_tablegen(CliftTypeInterfacesBasic.cpp.inc -gen-type-interface-defs)

set(LLVM_TARGET_DEFINITIONS CliftTypeInterfacesDefined.td)
mlir_tablegen(CliftTypeInterfacesDefined.h.inc -gen-type-interface-decls)
mlir_tablegen(CliftTypeInterfacesDefined.cpp.inc -gen-type-interface-defs)

set(LLVM_TARGET_DEFINITIONS CliftTypeInterfacesClass.td)
mlir_tablegen(CliftTypeInterfacesClass.h.inc -gen-type-interface-decls)
mlir_tablegen(CliftTypeInterfacesClass.cpp.inc -gen-type-interface-defs)

set(LLVM_TARGET_DEFINITIONS CliftAttrInterfaces.td)
mlir_tablegen(CliftAttrInterfaces.h.inc -gen-attr-interface-decls)
mlir_tablegen(CliftAttrInterfaces.cpp.inc -gen-attr-interface-defs)

set(LLVM_TARGET_DEFINITIONS CliftOpInterfaces.td)
mlir_tablegen(CliftOpInterfaces.h.inc -gen-op-interface-decls)
mlir_tablegen(CliftOpInterfaces.cpp.inc -gen-op-interface-defs)

add_mlir_dialect(Clift clift)
