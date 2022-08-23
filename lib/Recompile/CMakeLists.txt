#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

revng_add_analyses_library_internal(
  revngRecompile LinkForTranslationPipe.cpp LinkForTranslation.cpp
  CompileModulePipe.cpp)

target_link_libraries(revngRecompile revngSupport revngPipes ${LLVM_LIBRARIES})
