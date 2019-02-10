# Helper macro to create a new library containing analyses to be employed in
# revng-opt
macro(revng_add_analyses_library NAME EXPORT_NAME)
  add_library("${NAME}" SHARED ${ARGN})
  target_include_directories("${NAME}" INTERFACE $<INSTALL_INTERFACE:include/>)
  set_target_properties("${NAME}" PROPERTIES INSTALL_RPATH "\$ORIGIN/../../:\$ORIGIN/")
  install(TARGETS "${NAME}" EXPORT "${EXPORT_NAME}" LIBRARY DESTINATION lib/revng/analyses)
endmacro()

# This macro returns in ${RESULT} a list of files matching the pattern in the
# extra arguments. If the source is in a git repository, only tracked files are
# returned, otherwise a regular globbing expression is employed.
macro(git_ls_files_or_glob RESULT)

  execute_process(COMMAND git ls-files ${ARGN}
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    RESULT_VARIABLE GIT_LS_EXIT_CODE
    OUTPUT_VARIABLE GIT_LS_OUTPUT
    ERROR_VARIABLE GIT_LS_OUTPUT_STDERR)

  if(GIT_LS_EXIT_CODE EQUAL "0")
    string(REGEX REPLACE "\n" ";" ${RESULT} "${GIT_LS_OUTPUT}")
  else()
    file(GLOB_RECURSE ${RESULT} RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" ${ARGN})
  endif()

endmacro(git_ls_files_or_glob)

# This macro installs all the files matching the pattern in the extra arguments
macro(install_pattern)

  git_ls_files_or_glob(HEADERS_TO_INSTALL ${ARGN})

  file(RELATIVE_PATH RELATIVE_SOURCE_DIR ${CMAKE_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR})

  foreach(FILE ${HEADERS_TO_INSTALL})
    get_filename_component(INSTALL_PATH "${FILE}" DIRECTORY)
    install(FILES "${FILE}" DESTINATION ${RELATIVE_SOURCE_DIR}/${INSTALL_PATH})
  endforeach(FILE)

endmacro(install_pattern)
