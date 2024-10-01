#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

revng_add_executable(analyze Main.cpp)

target_link_libraries(analyze revngPipeline revngPipes)
