#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

revng_add_library_internal(revngRecompile SHARED LinkForTranslationPipe.cpp
                           LinkForTranslation.cpp)

target_link_libraries(revngRecompile revngSupport revngPipes ${LLVM_LIBRARIES})
