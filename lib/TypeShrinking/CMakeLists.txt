#
# This file is distributed under the MIT License. See LICENSE.mit for details.
#

revng_add_analyses_library_internal(revngTypeShrinking TypeShrinking.cpp
                                    BitLiveness.cpp DataFlowGraph.cpp)

target_link_libraries(revngTypeShrinking revngSupport)
