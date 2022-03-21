# Copyright rev.ng Srls. See LICENSE.md for details.
#

revng_add_analyses_library(
  revngcDataLayoutAnalysis
  revngc
  Frontend/DLACreateInterProceduralTypes.cpp
  Frontend/DLACreateIntraProceduralTypes.cpp
  Frontend/DLATypeSystemBuilder.cpp
  Frontend/SCEVBaseAddressExplorer.cpp
  Middleend/CollapseSCC.cpp
  Middleend/DLAComputeNonInterferingComponents.cpp
  Middleend/DLAComputeUpperMemberAccess.cpp
  Middleend/DLAPruneLayoutNodesWithoutLayout.cpp
  Middleend/DLARemoveTransitiveInheritanceEdges.cpp
  Middleend/DLAMakeInheritanceTree.cpp
  Middleend/DLACollapseSingleChild.cpp
  Middleend/DLADeduplicateUnionFields.cpp
  Middleend/DLARemoveConflictingEdges.cpp
  Middleend/DLAStep.cpp
  Middleend/FieldSizeComputation.cpp
  Middleend/RemoveBackedges.cpp
  Middleend/RemoveInvalidStrideEdges.cpp
  Middleend/SimplifyInstanceAtOffset0.cpp
  Backend/DLAMakeLayouts.cpp
  Backend/DLAMakeModelTypes.cpp
  Backend/DLAUpdateFuncPrototypes.cpp
  DLAModelFuncHelpers.cpp
  DLALayouts.cpp
  DLAPass.cpp
  DLATypeSystem.cpp)

target_link_libraries(revngcDataLayoutAnalysis revngcSupport revng::revngModel
                      revng::revngSupport ${LLVM_LIBRARIES})
