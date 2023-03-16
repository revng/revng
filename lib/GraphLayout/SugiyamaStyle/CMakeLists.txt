#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

revng_add_library_internal(
  revngSugiyamaGraphLayout
  SHARED
  Compute.cpp
  EdgeRouting.cpp
  GraphPreparation.cpp
  HorizontalPositions.cpp
  LaneDistribution.cpp
  LayoutConversion.cpp
  LinearSegmentSelection.cpp
  NodeRanking.cpp
  PermutationSelection.cpp
  TopologicalOrdering.cpp
  VerticalPositions.cpp)

target_link_libraries(revngSugiyamaGraphLayout revngSupport ${LLVM_LIBRARIES})
