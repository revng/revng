#
# Copyright rev.ng Srls. See LICENSE.md for details.
#

revng_add_analyses_library(revngcIRCanonicalization
	revngc
	AddIRSerializationMarkerPass.cpp
	MakeModelGEPPass.cpp
	MarkerFunctions.cpp
	ExitSSAPass.cpp)

target_link_libraries(revngcIRCanonicalization
	revngcDecompiler
	revngcLivenessAnalysis
	revngcMarkForSerialization
	revngcTargetFunctionOption
	revng::revngModel
	revng::revngSupport
	${LLVM_LIBRARIES})

