#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

revng_add_library_internal(
  revngCliftTransforms
  SHARED
  BooleanNegations.cpp
  Expressions.cpp
  LabelMerging.cpp
  Legalization.cpp
  LoopDetection.cpp
  PatternRewriter.cpp
  RewriteHelpers.cpp
  Statements.cpp
  TerminalBranchComplementHoisting.cpp
  TrivialReturnElimination.cpp)

target_link_libraries(revngCliftTransforms MLIRTransformUtils revngClift)

add_dependencies(revngCliftTransforms MLIRCliftTransformsIncGen)
