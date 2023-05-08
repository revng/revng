#
# Copyright rev.ng Labs Srl. See LICENSE.md for details.
#

add_mlir_dialect_library(
  MLIRCliftDialect
  CliftOps.cpp
  Clift.cpp
  CliftTypes.cpp
  CliftAttributes.cpp
  CliftEnums.cpp
  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/Clift
  DEPENDS
  LINK_LIBS
  PUBLIC
  MLIRDialect
  MLIRIR)
