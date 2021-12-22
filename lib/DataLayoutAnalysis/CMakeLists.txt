# Copyright rev.ng Srls. See LICENSE.md for details.
#

revng_add_analyses_library(revngcDataLayoutAnalysis revngc
  Frontend/DLACreateInterProceduralTypes.cpp
  Frontend/DLACreateIntraProceduralTypes.cpp
  Frontend/DLATypeSystemBuilder.cpp
  Frontend/SCEVBaseAddressExplorer.cpp
  Middleend/DLACollapseIdentityAndInheritanceCC.cpp
  Middleend/DLAComputeNonInterferingComponents.cpp
  Middleend/DLAComputeUpperMemberAccess.cpp
  Middleend/DLAPruneLayoutNodesWithoutLayout.cpp
  Middleend/DLARemoveTransitiveInheritanceEdges.cpp
  Middleend/DLAMakeInheritanceTree.cpp
  Middleend/DLACollapseSingleChild.cpp
  Middleend/DLADeduplicateUnionFields.cpp
  Middleend/DLARemoveConflictingEdges.cpp
  Middleend/DLAStep.cpp
  Backend/DLAMakeLayouts.cpp
  Backend/DLAMakeModelTypes.cpp
  Backend/DLAUpdateFuncPrototypes.cpp
  DLAHelpers.cpp
  DLAModelFuncHelpers.cpp
  DLALayouts.cpp
  DLAPass.cpp
  DLATypeSystem.cpp)

target_link_libraries(revngcDataLayoutAnalysis
  revngcSupport
  revng::revngModel
  revng::revngSupport
  ${LLVM_LIBRARIES})
