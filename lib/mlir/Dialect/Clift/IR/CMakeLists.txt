#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

add_mlir_dialect_library(
  MLIRCliftDialect
  CliftOps.cpp
  Clift.cpp
  CliftTypes.cpp
  CliftAttributes.cpp
  CliftInterfaces.cpp
  CliftEnums.cpp
  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/Clift
  DEPENDS
  LINK_LIBS
  PUBLIC
  MLIRTransforms
  MLIRDialect
  MLIRIR
  PRIVATE
  revng::revngModel)
