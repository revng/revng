#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

add_mlir_dialect_library(
  MLIRCliftDialect
  SHARED
  CliftDialect.cpp
  Clift.cpp
  CliftOpInterfaces.cpp
  CliftOpTraits.cpp
  CliftTypes.cpp
  CliftAttributes.cpp
  CliftInterfaces.cpp
  CliftEnums.cpp
  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/Clift
  LINK_LIBS
  revngSupport
  PUBLIC
  MLIRTransforms
  MLIRDialect
  MLIRIR)

revng_register_library(MLIRCliftDialect revng)
