#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

revng_add_executable(revng-pipeline Main.cpp)

target_link_libraries(revng-pipeline revngPipeline revngPipes revngRecompile
                      revngEarlyFunctionAnalysis)

add_subdirectory(invalidate)
