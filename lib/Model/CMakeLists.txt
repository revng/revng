#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

revng_add_analyses_library_internal(revngModel
  Binary.cpp
  TupleTreeDiff.cpp
  LoadModelPass.cpp
  SerializeModelPass.cpp)

target_link_libraries(revngModel
  revngSupport)
