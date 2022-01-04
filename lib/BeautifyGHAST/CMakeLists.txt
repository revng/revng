#
# Copyright rev.ng Srls. See LICENSE.md for details.
#

revng_add_analyses_library(revngcBeautifyGHAST
	revngc
	BeautifyGHASTPass.cpp
	BeautifyGHAST.cpp)

target_link_libraries(revngcBeautifyGHAST
	revngcRestructureCFGPass
	revngcTargetFunctionOption
	revngcSupport
	revng::revngSupport
	${LLVM_LIBRARIES})
