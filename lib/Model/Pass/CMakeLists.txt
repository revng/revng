#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

revng_add_library_internal(
  revngModelPasses
  SHARED
  AddPrimitiveTypes.cpp
  AddPrimitiveTypesAnalysis.cpp
  DeduplicateEquivalentTypes.cpp
  FixModel.cpp
  OptionCategory.cpp
  PromoteOriginalName.cpp
  PurgeUnnamedAndUnreachableTypes.cpp
  Verify.cpp)

target_link_libraries(revngModelPasses revngModel revngPipes)
