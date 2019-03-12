revng_add_analyses_library(PHIASAPAssignmentInfo revngc
                           PHIASAPAssignmentInfo.cpp)

target_link_libraries(PHIASAPAssignmentInfo
                      revng::revngSupport
                      ${LLVM_LIBRARIES})

