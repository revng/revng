#
# Copyright (c) rev.ng Labs Srl. See LICENSE.md for details.
#

revng_add_executable(revng-model-to-header Main.cpp)

target_link_libraries(revng-model-to-header revngcSupport revng::revngModel
                      revng::revngSupport ${LLVM_LIBRARIES} revngcModelToHeader)
