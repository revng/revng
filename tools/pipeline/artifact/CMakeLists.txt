#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

revng_add_executable(artifact Main.cpp)

target_link_libraries(artifact revngPipeline revngPipes)
