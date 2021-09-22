#
# Copyright rev.ng Srls. See LICENSE.md for details.
#

revng_add_executable(revng-dump-schema DumpSchemaMain.cpp)

target_link_libraries(revng-dump-schema
  revngModel
  revngSupport
  ${LLVM_LIBRARIES}
  )
