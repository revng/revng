#
# Copyright (c) rev.ng Labs Srl. See LICENSE.md for details.
#

revng_add_executable(revng-decompile Main.cpp)

target_link_libraries(revng-decompile revngcBackend revng::revngModel
                      revng::revngPipes revng::revngSupport ${LLVM_LIBRARIES})
