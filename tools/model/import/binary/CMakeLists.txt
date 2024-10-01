#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

revng_add_executable(model-import-binary Main.cpp)

target_link_libraries(model-import-binary revngModelImporterBinary)
