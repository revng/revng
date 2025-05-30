#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

add_mlir_dialect_library(
  MLIRCliftTransforms
  SHARED
  Beautify/BranchEqualization.cpp
  Beautify/ExpressionRootSimplification.cpp
  Beautify/Expressions.cpp
  Beautify/ImmediateRadixDeduction.cpp
  Beautify/LoopDetection.cpp
  Beautify/Statements.cpp
  Beautify/ReturnConversion.cpp
  Beautify.cpp
  CSemantics.cpp
  DEPENDS
  LINK_LIBS
  PUBLIC
  MLIRCliftDialect
  MLIRCliftUtils)

revng_register_library(MLIRCliftTransforms revng)
