#
# Copyright (c) rev.ng Srls. See LICENSE.md for details.
#

revng_add_executable(revng-helpers-to-header Main.cpp)

target_link_libraries(
  revng-helpers-to-header revng::revngModel revng::revngSupport
  ${LLVM_LIBRARIES} revngcHelpersToHeader)
