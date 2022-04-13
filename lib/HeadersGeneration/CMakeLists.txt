#
# Copyright rev.ng Labs Srl. See LICENSE.md for details.
#

# ModelToHeader

revng_add_analyses_library(revngcModelToHeader revngc ModelToHeader.cpp
                           DependencyGraph.cpp)

target_link_libraries(revngcModelToHeader revng::revngModel revng::revngSupport
                      revngcTypeNames ${LLVM_LIBRARIES})

revng_add_analyses_library(revngcModelToHeaderPass revngc ModelToHeaderPass.cpp)

target_link_libraries(revngcModelToHeaderPass revng::revngModel
                      revng::revngSupport ${LLVM_LIBRARIES} revngcModelToHeader)

# HelpersToHeader

revng_add_analyses_library(revngcHelpersToHeader revngc HelpersToHeader.cpp)

target_link_libraries(
  revngcHelpersToHeader revng::revngModel revng::revngSupport revngcSupport
  revngcTypeNames ${LLVM_LIBRARIES})

revng_add_analyses_library(revngcHelpersToHeaderPass revngc
                           HelpersToHeaderPass.cpp)

target_link_libraries(
  revngcHelpersToHeaderPass revng::revngModel revng::revngSupport revngcSupport
  ${LLVM_LIBRARIES} revngcHelpersToHeader)
