#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

revng_add_library_internal(
  revngModelPasses
  SHARED
  DeduplicateEquivalentTypes.cpp
  OptionCategory.cpp
  PromoteOriginalName.cpp
  PurgeUnnamedAndUnreachableTypes.cpp
  Verify.cpp
  Visibility.cpp)

target_link_libraries(revngModelPasses revngModel revngPipes)
