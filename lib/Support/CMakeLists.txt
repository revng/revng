#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

revng_add_library_internal(
  revngSupport
  SHARED
  ProgramRunner.cpp
  Assert.cpp
  BasicBlockID.cpp
  CommandLine.cpp
  CommonOptions.cpp
  Debug.cpp
  ExplicitSpecializations.cpp
  IRAnnotators.cpp
  FunctionTags.cpp
  IRHelpers.cpp
  LDDTree.cpp
  MetaAddress.cpp
  ModuleStatistics.cpp
  OnQuit.cpp
  OriginalAssemblyAnnotationWriter.cpp
  PathList.cpp
  Progress.cpp
  ProgramCounterHandler.cpp
  ResourceFinder.cpp
  SelfReferencingDbgAnnotationWriter.cpp
  Statistics.cpp
  Tag.cpp
  GzipTarFile.cpp
  GzipStream.cpp)

llvm_map_components_to_libnames(LLVM_LIBRARIES Support Core Object)

include(FindLibArchive)

if(NOT LibArchive_FOUND)
  message(FATAL_ERROR "libarchive not found")
endif()

target_link_libraries(revngSupport z ${LibArchive_LIBRARIES} ${LLVM_LIBRARIES})

target_include_directories(revngSupport INTERFACE $<INSTALL_INTERFACE:include/>)

# revngSupport has a dependency to the generated model files because of revng.h
#
# revng.h includes model headers which depends on the generated headers. we
# cannot add the dependency to the model because the model itself depends on
# support and would generate a cyclic dependency.
#
# TODO: drop dependency from model in revng.h
add_dependencies(revngSupport generate-revngModel-tuple-tree-code)
