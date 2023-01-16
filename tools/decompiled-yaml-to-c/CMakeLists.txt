#
# Copyright (c) rev.ng Labs Srl. See LICENSE.md for details.
#

revng_add_executable(revng-decompiled-yaml-to-c Main.cpp)

target_link_libraries(revng-decompiled-yaml-to-c revngcBackend revngcSupport
                      revng::revngPipes revng::revngSupport ${LLVM_LIBRARIES})
