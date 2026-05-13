#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

# The inline-helpers chain (link / inline / delete), exposed both as LLVM passes
# and as free functions. It lives in its own library so that
# EarlyFunctionAnalysis can depend on it without creating a cycle with
# FunctionIsolation, which depends on EarlyFunctionAnalysis.
revng_add_analyses_library_internal(
  revngInlineHelpers DeleteHelperBodies.cpp InlineHelpers.cpp
  LinkHelpersToInline.cpp PostInlineHelpersVerifyPass.cpp)

llvm_map_components_to_libnames(LLVM_LIBRARIES TransformUtils)

target_link_libraries(revngInlineHelpers revngBasicAnalyses revngModel
                      revngSupport ${LLVM_LIBRARIES})
