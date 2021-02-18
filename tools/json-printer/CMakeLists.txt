#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

revng_add_executable(json-printer
  Main.cpp
  JSONCustom.cpp)
target_link_libraries(json-printer
  revngSupport
  ${LLVM_LIBRARIES})
