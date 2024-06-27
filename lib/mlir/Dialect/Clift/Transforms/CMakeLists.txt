#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

# TODO: Use revng_add_library instead.
add_mlir_dialect_library(
  MLIRCliftTransforms
  ModelOption.cpp
  DEPENDS
  LINK_LIBS
  PUBLIC
  MLIRCliftDialect
  MLIRCliftUtils)
