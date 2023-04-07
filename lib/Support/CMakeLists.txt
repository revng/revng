#
# Copyright rev.ng Labs Srl. See LICENSE.md for details.
#

revng_add_analyses_library(revngcSupport revngc FunctionTags.cpp IRHelpers.cpp
                           ModelHelpers.cpp SimplifyCFGWithHoistAndSinkPass.cpp)

target_link_libraries(revngcSupport revng::revngEarlyFunctionAnalysis
                      revng::revngABI revng::revngModel revng::revngSupport)
