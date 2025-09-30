#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

add_mlir_dialect_library(
  MLIRCliftTransforms
  SHARED
  CSemantics.cpp
  CBackend.cpp
  DEPENDS
  LINK_LIBS
  PUBLIC
  MLIRCliftDialect
  MLIRCliftUtils)

revng_register_library(MLIRCliftTransforms revng)
