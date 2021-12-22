#
# Copyright rev.ng Srls. See LICENSE.md for details.
#

revng_add_analyses_library(revngcUtils revngc Utils.cpp)

target_link_libraries(revngcUtils
	revng::revngModel
	revng::revngSupport
	${LLVM_LIBRARIES})
