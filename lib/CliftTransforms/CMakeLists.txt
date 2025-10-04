#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

revng_add_library_internal(revngCliftTransforms SHARED Legalization.cpp)

target_link_libraries(revngCliftTransforms PUBLIC MLIRTransformUtils revngClift)

add_dependencies(revngCliftTransforms MLIRCliftTransformsIncGen)

revng_add_analyses_library_internal(revngCliftTransformsAnalyses
                                    LegalizationPipe.cpp)

target_link_libraries(revngCliftTransformsAnalyses PUBLIC revngCliftTransforms
                                                          revngCliftContainer)
