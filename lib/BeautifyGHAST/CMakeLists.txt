#
# Copyright rev.ng Srls. See LICENSE.md for details.
#

revng_add_analyses_library(BeautifyGHAST
	revngc
	BeautifyGHASTPass.cpp
	BeautifyGHAST.cpp)

target_link_libraries(BeautifyGHAST
	MarkForSerialization
	RestructureCFGPass
	TargetFunctionOption
	Utils
	revng::revngSupport
	${LLVM_LIBRARIES})
