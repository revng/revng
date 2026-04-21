#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

revng_add_library_internal(
  revngCliftTransforms
  SHARED
  BestTraversal.cpp
  BooleanNegations.cpp
  EmitFieldAccesses.cpp
  Expressions.cpp
  FieldAccessReplacement.cpp
  ImmediateRadixDeduction.cpp
  ImplicitCastElision.cpp
  LabelMerging.cpp
  Legalization.cpp
  LoopDetection.cpp
  PatternRewriter.cpp
  PointerArithmetic.cpp
  RewriteHelpers.cpp
  Statements.cpp
  TerminalBranchComplementHoisting.cpp
  TrivialReturnElimination.cpp)

target_link_libraries(revngCliftTransforms MLIRTransformUtils revngClift)

add_dependencies(revngCliftTransforms MLIRCliftTransformsIncGen)
