#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

revng_add_library_internal(
  revngCliftTransforms
  SHARED
  BooleanCastElimination.cpp
  BooleanNegations.cpp
  EmitFieldAccesses.cpp
  EmitFieldAccesses/BestTraversal.cpp
  EmitFieldAccesses/FieldAccessReplacement.cpp
  EmitFieldAccesses/PointerArithmetic.cpp
  Expressions.cpp
  ExpressionPostProcessing.cpp
  ImmediateRadixDeduction.cpp
  ImplicitCastElision.cpp
  LabelMerging.cpp
  Legalization.cpp
  LoopDetection.cpp
  PatternRewriter.cpp
  PromoteBreakContinue.cpp
  RewriteHelpers.cpp
  Statements.cpp
  TerminalBranchComplementHoisting.cpp
  TightenVariableScopes.cpp
  TrivialReturnElimination.cpp
  TypeRefinement.cpp
  TypeRefinementWithEFA.cpp
  VariableInitializerHoisting.cpp)

target_link_libraries(revngCliftTransforms MLIRTransformUtils revngClift)

add_dependencies(revngCliftTransforms MLIRCliftTransformsIncGen)
