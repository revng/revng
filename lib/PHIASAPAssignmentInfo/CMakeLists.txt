#
# Copyright rev.ng Labs Srl. See LICENSE.md for details.
#

revng_add_analyses_library(revngcPHIASAPAssignmentInfo revngc
                           PHIASAPAssignmentInfo.cpp)

target_link_libraries(revngcPHIASAPAssignmentInfo revng::revngModel
                      revng::revngSupport ${LLVM_LIBRARIES})
