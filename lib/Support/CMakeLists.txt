#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

revng_add_library_internal(
  revngSupport
  SHARED
  CTarget.cpp
  ProgramRunner.cpp
  Assert.cpp
  BasicBlockID.cpp
  CommandLine.cpp
  CommonOptions.cpp
  CustomizedLLVMPasses.cpp
  Debug.cpp
  ExplicitSpecializations.cpp
  InitRevng.cpp
  IRHelpers.cpp
  LDDTree.cpp
  MetaAddress.cpp
  ModuleStatistics.cpp
  OnQuit.cpp
  PathList.cpp
  Progress.cpp
  ResourceFinder.cpp
  Statistics.cpp
  Tag.cpp
  GzipTarFile.cpp
  Tar.cpp
  GzipStream.cpp
  ZstdStream.cpp)

include(FindLibArchive)

if(NOT LibArchive_FOUND)
  message(FATAL_ERROR "libarchive not found")
endif()

llvm_map_components_to_libnames(
  LLVM_LIBRARIES
  Analysis
  CodeGen
  Core
  GlobalISel
  InstCombine
  IPO
  IRReader
  Linker
  Object
  Passes
  ScalarOpts
  Support
  Target
  TransformUtils
  Vectorize
  X86AsmParser
  X86CodeGen
  X86Desc
  X86Info)
target_link_libraries(revngSupport z zstd ${LibArchive_LIBRARIES}
                      ${LLVM_LIBRARIES})

target_include_directories(revngSupport INTERFACE $<INSTALL_INTERFACE:include/>)

# revngSupport has a dependency to the generated model files because of revng.h
#
# revng.h includes model headers which depends on the generated headers. we
# cannot add the dependency to the model because the model itself depends on
# support and would generate a cyclic dependency.
#
# TODO: drop dependency from model in revng.h
add_dependencies(revngSupport generate-revngModel-tuple-tree-code)
