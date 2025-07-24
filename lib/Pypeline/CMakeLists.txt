#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

add_subdirectory(TraceRunner)
revng_add_library_internal(revngPypeline SHARED Registry.cpp)
target_link_libraries(revngPypeline revngSupport)
