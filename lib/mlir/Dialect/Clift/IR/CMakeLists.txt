#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

# TODO: Use revng_add_library instead.
add_mlir_dialect_library(
  MLIRCliftDialect
  SHARED
  CliftOps.cpp
  Clift.cpp
  CliftOpInterfaces.cpp
  CliftOpTraits.cpp
  CliftTypes.cpp
  CliftAttributes.cpp
  CliftInterfaces.cpp
  CliftEnums.cpp
  CliftParser.cpp
  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/Clift
  DEPENDS
  revngModel
  LINK_LIBS
  revngModel
  revngSupport
  PUBLIC
  MLIRTransforms
  MLIRDialect
  MLIRIR)

install(
  TARGETS MLIRCliftDialect
  EXPORT revng
  LIBRARY DESTINATION lib/
  ARCHIVE DESTINATION lib/)
