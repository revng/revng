#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

revng_add_executable(invalidate Main.cpp)

target_link_libraries(invalidate revngPipeline revngPipes)
