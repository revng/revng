#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

revng_add_executable(model-import-debug-info Main.cpp)

target_link_libraries(model-import-debug-info revngABI revngModel
                      revngModelImporterBinary revngModelImporterDebugInfo)
