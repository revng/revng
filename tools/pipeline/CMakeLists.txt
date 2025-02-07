#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

revng_add_executable(pipeline Main.cpp)

target_link_libraries(pipeline revngPipeline revngPipes)

add_subdirectory(invalidate)
add_subdirectory(artifact)
add_subdirectory(analyze)
add_subdirectory(pipe)
add_subdirectory(init)
