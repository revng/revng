#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

revng_add_analyses_library_internal(revngDump
  CollectCFG.cpp
  CollectNoreturn.cpp)

target_link_libraries(revngDump
  revngSupport)
