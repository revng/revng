#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

add_library(revngUnitTestHelpers
  STATIC
  DotGraphObject.cpp)

target_link_libraries(revngUnitTestHelpers
  revngSupport ${LLVM_LIBRARIES})

target_include_directories(revngUnitTestHelpers
  INTERFACE $<INSTALL_INTERFACE:include/>)

install(TARGETS revngUnitTestHelpers
  EXPORT revng
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib)
