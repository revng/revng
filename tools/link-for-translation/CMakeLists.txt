#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

revng_add_executable(revng-link-for-translation Main.cpp)

target_link_libraries(revng-link-for-translation revngRecompile)
