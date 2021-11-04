#
# Copyright rev.ng Srls. See LICENSE.md for details.
#

revng_add_analyses_library(RestructureCFGPass
	revngc
	ASTNode.cpp
	ASTTree.cpp
	BasicBlockNode.cpp
	ExprNode.cpp
	MetaRegion.cpp
	RegionCFGTree.cpp
	RestructureCFG.cpp)

target_link_libraries(RestructureCFGPass
	revng::revngModel
	revng::revngSupport
	TargetFunctionOption
	${LLVM_LIBRARIES})
