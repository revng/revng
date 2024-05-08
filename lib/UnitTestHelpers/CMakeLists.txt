#
# This file is distributed under the MIT License. See LICENSE.mit for details.
#

revng_add_library_internal(revngUnitTestHelpers SHARED DotGraphObject.cpp)

target_link_libraries(revngUnitTestHelpers revngSupport ${LLVM_LIBRARIES})

target_include_directories(revngUnitTestHelpers
                           INTERFACE $<INSTALL_INTERFACE:include/>)
