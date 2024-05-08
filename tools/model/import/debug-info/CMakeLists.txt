#
# This file is distributed under the MIT License. See LICENSE.mit for details.
#

revng_add_executable(revng-model-import-debug-info Main.cpp)

target_link_libraries(revng-model-import-debug-info revngABI revngModel
                      revngModelImporterBinary revngModelImporterDebugInfo)
