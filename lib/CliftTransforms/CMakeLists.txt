#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

revng_add_library_internal(revngCliftTransforms SHARED Legalization.cpp
                           LoopDetection.cpp)

target_link_libraries(revngCliftTransforms MLIRTransformUtils revngClift)

add_dependencies(revngCliftTransforms MLIRCliftTransformsIncGen)
