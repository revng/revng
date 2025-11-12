#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

revng_add_library_internal(
  revngClift
  SHARED
  Clift.cpp
  CliftAttributes.cpp
  CliftDialect.cpp
  CliftEnums.cpp
  CliftInterfaces.cpp
  CliftOpHelpers.cpp
  CliftOpInterfaces.cpp
  CliftOpTraits.cpp
  CliftTypes.cpp)

target_link_libraries(
  revngClift
  MLIRBytecodeReader
  MLIRBytecodeWriter
  MLIRDialect
  MLIRInferTypeOpInterface
  MLIRIR
  MLIRParser
  revngSupport)

add_dependencies(revngClift MLIRCliftIncGen)
