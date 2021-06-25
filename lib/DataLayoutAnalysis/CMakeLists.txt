# Copyright rev.ng Srls. See LICENSE.md for details.
#

revng_add_analyses_library(DataLayoutAnalysis revngc
  Frontend/DLACreateInterProceduralTypes.cpp
  Frontend/DLACreateIntraProceduralTypes.cpp
  Frontend/DLATypeSystemBuilder.cpp
  Middleend/DLACollapseIdentityAndInheritanceCC.cpp
  Middleend/DLAComputeNonInterferingComponents.cpp
  Middleend/DLAComputeUpperMemberAccess.cpp
  Middleend/DLAPruneLayoutNodesWithoutLayout.cpp
  Middleend/DLARemoveTransitiveInheritanceEdges.cpp
  Middleend/DLAMakeInheritanceTree.cpp
  Middleend/DLACollapseSingleChild.cpp
  Middleend/DLAStep.cpp
  Backend/DLAMakeLayouts.cpp
  DLAHelpers.cpp
  DLALayouts.cpp
  DLAPass.cpp
  DLATypeSystem.cpp)

target_link_libraries(DataLayoutAnalysis
  revng::revngModel
  revng::revngSupport
  ${LLVM_LIBRARIES})