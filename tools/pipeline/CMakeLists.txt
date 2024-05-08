#
# This file is distributed under the MIT License. See LICENSE.mit for details.
#

revng_add_executable(revng-pipeline Main.cpp)

target_link_libraries(revng-pipeline revngPipeline revngPipes)

add_subdirectory(invalidate)
add_subdirectory(artifact)
add_subdirectory(analyze)
