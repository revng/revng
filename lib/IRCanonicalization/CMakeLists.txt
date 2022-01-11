#
# Copyright rev.ng Srls. See LICENSE.md for details.
#

revng_add_analyses_library(IRCanonicalization
	revngc
	AddIRSerializationMarkerPass.cpp)

target_link_libraries(IRCanonicalization
	revngcLivenessAnalysis
	revngcMarkForSerialization
	revngcTargetFunctionOption
	revng::revngSupport
	${LLVM_LIBRARIES})

