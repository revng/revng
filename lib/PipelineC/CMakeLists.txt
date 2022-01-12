#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

revng_add_library_internal(revngPipelineC SHARED
	PipelineC.cpp
	)

target_link_libraries(revngPipelineC
  revngPipes
  ${LLVM_LIBRARIES})
