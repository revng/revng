#
# Copyright rev.ng Srls. See LICENSE.md for details.
#

revng_add_analyses_library(MarkForSerialization
	revngc
	MarkForSerializationPass.cpp)

target_link_libraries(MarkForSerialization
	LivenessAnalysis
	RestructureCFGPass
	Support
	TargetFunctionOption
	revng::revngSupport
	${LLVM_LIBRARIES})
