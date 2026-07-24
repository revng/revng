#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

revng_add_analyses_library_internal(
  revngRecompile LinkForTranslation.cpp OriginalAssemblyAnnotationWriter.cpp
  CompileModulePipe.cpp)

llvm_map_components_to_libnames(LLVM_LIBRARIES Core Support MC)
target_link_libraries(revngRecompile revngLift revngModelImporterBinary
                      revngSupport ${LLVM_LIBRARIES})
