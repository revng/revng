#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

revng_add_library_internal(revngPipeline SHARED
  LLVMPipe.cpp
  Errors.cpp
  Target.cpp
  Context.cpp
  Step.cpp
  Kind.cpp
  Contract.cpp
  ContainerSet.cpp
  Runner.cpp
  Loader.cpp
  LLVMKind.cpp
  LLVMContainer.cpp)

target_link_libraries(revngPipeline
  revngSupport
  ${LLVM_LIBRARIES})
