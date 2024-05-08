#
# This file is distributed under the MIT License. See LICENSE.mit for details.
#

#
# General docs
#
set(MAN_PAGES)
set(DOC_HTML FromIRToExecutable.rst GeneratedIRReference.rst Overview.rst
             PythonExample.rst)
set(DOC_COPY instrument.py)

set(DOC_DEPS)

find_program(RST2MAN rst2man.py)
if(NOT "${RST2MAN}" STREQUAL RST2MAN-NOTFOUND)
  foreach(INPUT_FILE ${MAN_PAGES})
    string(REPLACE ".rst" ".1" OUTPUT_FILE "${INPUT_FILE}")
    add_custom_command(
      OUTPUT "${OUTPUT_FILE}"
      COMMAND "${RST2MAN}" "${CMAKE_CURRENT_SOURCE_DIR}/${INPUT_FILE}" >
              "${OUTPUT_FILE}"
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
    add_custom_command(
      OUTPUT "${OUTPUT_FILE}"
      COMMAND "${RST2HTML}" "${CMAKE_CURRENT_SOURCE_DIR}/${INPUT_FILE}" >
              "${OUTPUT_FILE}"
      MAIN_DEPENDENCY "${INPUT_FILE}")
    set(DOC_DEPS ${DOC_DEPS} ${OUTPUT_FILE})
    install(FILES ${CMAKE_CURRENT_BINARY_DIR}/${OUTPUT_FILE}
            DESTINATION ${CMAKE_INSTALL_PREFIX}/share/doc/revng/)
  endforeach()
endif()

foreach(INPUT_FILE ${DOC_COPY})
  configure_file("${CMAKE_CURRENT_SOURCE_DIR}/${INPUT_FILE}" "${INPUT_FILE}"
                 COPYONLY)
  install(FILES ${CMAKE_CURRENT_BINARY_DIR}/${INPUT_FILE}
          DESTINATION ${CMAKE_INSTALL_PREFIX}/share/doc/revng/)
endforeach()

add_custom_target(docs ALL DEPENDS ${DOC_DEPS})

#
# Doxygen
#
find_package(Doxygen)
if(DOXYGEN_FOUND)
  execute_process(
    COMMAND git ls-files
    WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
    RESULT_VARIABLE GIT_LS_EXIT_CODE
    OUTPUT_VARIABLE GIT_LS_OUTPUT
    ERROR_VARIABLE GIT_LS_OUTPUT_STDERR)

  if(GIT_LS_EXIT_CODE EQUAL "0")
    string(REGEX REPLACE "\n" ";" GIT_LS_OUTPUT "${GIT_LS_OUTPUT}")
    set(DOXYGEN_INPUTS "")
    foreach(FILE ${GIT_LS_OUTPUT})
      set(DOXYGEN_INPUTS "${DOXYGEN_INPUTS} ${CMAKE_SOURCE_DIR}/${FILE}")
    endforeach(FILE)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile.in
                   ${CMAKE_BINARY_DIR}/Doxyfile @ONLY)
    add_custom_target(
      doc
      ${DOXYGEN_EXECUTABLE} ${CMAKE_BINARY_DIR}/Doxyfile
      WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
      COMMENT "Generating API documentation with Doxygen"
      VERBATIM)
  else()
    message(
      WARNING
        "Source directory is not a git repository, disabling Doxygen. Error was: ${GIT_LS_OUTPUT_STDERR}"
    )
  endif()

endif(DOXYGEN_FOUND)
