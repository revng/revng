#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

revng_add_library_internal(
  revngModelPasses
  SHARED
  DeduplicateCollidingNames.cpp
  DeduplicateEquivalentTypes.cpp
  FlattenPrimitiveTypedefs.cpp
  OptionCategory.cpp
  PurgeUnnamedAndUnreachableTypes.cpp
  Verify.cpp)

target_link_libraries(revngModelPasses revngModel revngPipes)
