#
# This file is distributed under the MIT License. See LICENSE.md for details.
#
# Check that required python dependencies are available

file(STRINGS "requirements.txt"
    REQUIREMENTS
    REGEX "^[^#].+$"
)

execute_process(
    COMMAND pip3 list
    RESULT_VARIABLE RETURNCODE
    OUTPUT_VARIABLE PYTHON_PACKAGES
    ERROR_FILE "/dev/null"
)

foreach(REQUIREMENT ${REQUIREMENTS})
  string(REGEX MATCH "\n${REQUIREMENT} " RESULT "\n${PYTHON_PACKAGES}")
  if("${RESULT}" STREQUAL "")
    message(FATAL_ERROR "Python requirement missing: ${REQUIREMENT}")
  endif()
endforeach()

# Check pyparsing version
string(REGEX MATCH "\npyparsing *2.4.7\n" RESULT "\n${PYTHON_PACKAGES}\n")
if("${RESULT}" STREQUAL "")
  message(FATAL_ERROR "pyparsing version 2.4.7 required. 3.x versions are known to cause build issues")
endif()
