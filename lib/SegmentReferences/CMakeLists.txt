#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

revng_add_analyses_library_internal(
  revngSegmentReferences DetectCStrings.cpp EmitFunctionPointers.cpp
  EmitSegmentReferences.cpp EmitStringConstants.cpp SegmentUsesEnumerator.cpp)

target_link_libraries(revngSegmentReferences revngSupport revngModel revngABI
                      ${LLVM_LIBRARIES})
