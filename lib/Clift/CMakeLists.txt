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
  CliftOpInterfaces.cpp
  CliftOpTraits.cpp
  CliftTypes.cpp)

target_link_libraries(
  revngClift
  PUBLIC MLIRBytecodeReader
         MLIRBytecodeWriter
         MLIRDialect
         MLIRInferTypeOpInterface
         MLIRIR
         MLIRParser
         revngSupport)

add_dependencies(revngClift MLIRCliftIncGen)

revng_add_analyses_library_internal(revngCliftContainer CliftContainer.cpp)

target_link_libraries(revngCliftContainer PUBLIC revngClift revngPipeline
                                                 revngPipes)
