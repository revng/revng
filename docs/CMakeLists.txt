#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

set(MAN_PAGES RevambDumpUsage.rst RevambUsage.rst TranslateUsage.rst)
set(DOC_HTML BuildSystem.rst FromIRToExecutable.rst GeneratedIRReference.rst
  PythonExample.rst)
set(DOC_COPY instrument.py)

set(DOC_DEPS)

find_program(RST2MAN rst2man.py)
if(NOT "${RST2MAN}" STREQUAL RST2MAN-NOTFOUND)
  foreach(INPUT_FILE ${MAN_PAGES})
    string(REPLACE ".rst" ".1" OUTPUT_FILE "${INPUT_FILE}")
    add_custom_command(OUTPUT "${OUTPUT_FILE}"
      COMMAND "${RST2MAN}" "${CMAKE_CURRENT_SOURCE_DIR}/${INPUT_FILE}" > "${OUTPUT_FILE}"
      MAIN_DEPENDENCY "${INPUT_FILE}")
    set(DOC_DEPS ${DOC_DEPS} ${OUTPUT_FILE})
    install(FILES ${CMAKE_CURRENT_BINARY_DIR}/${OUTPUT_FILE}
      DESTINATION ${CMAKE_INSTALL_PREFIX}/share/man/man1)
  endforeach()
endif()

find_program(RST2HTML rst2html.py)
if(NOT "${RST2HTML}" STREQUAL RST2HTML-NOTFOUND)
  foreach(INPUT_FILE ${DOC_HTML})
    string(REPLACE ".rst" ".html" OUTPUT_FILE "${INPUT_FILE}")
    add_custom_command(OUTPUT "${OUTPUT_FILE}"
      COMMAND "${RST2HTML}" "${CMAKE_CURRENT_SOURCE_DIR}/${INPUT_FILE}" > "${OUTPUT_FILE}"
      MAIN_DEPENDENCY "${INPUT_FILE}")
    set(DOC_DEPS ${DOC_DEPS} ${OUTPUT_FILE})
    install(FILES ${CMAKE_CURRENT_BINARY_DIR}/${OUTPUT_FILE}
      DESTINATION ${CMAKE_INSTALL_PREFIX}/share/doc/revng/)
  endforeach()
endif()

foreach(INPUT_FILE ${DOC_COPY})
  configure_file("${CMAKE_CURRENT_SOURCE_DIR}/${INPUT_FILE}" "${INPUT_FILE}" COPYONLY)
  install(FILES ${CMAKE_CURRENT_BINARY_DIR}/${INPUT_FILE}
    DESTINATION ${CMAKE_INSTALL_PREFIX}/share/doc/revng/)
endforeach()

add_custom_target(docs ALL DEPENDS ${DOC_DEPS})
