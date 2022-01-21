#
# This file is distributed under the MIT License. See LICENSE.md for details.
#
# Check that required python dependencies are available

file(STRINGS "requirements.txt"
    REQUIREMENTS
    REGEX "^[^#].+$"
    )

foreach(REQUIREMENT ${REQUIREMENTS})
  execute_process(
      COMMAND pip3 show "${REQUIREMENT}"
      RESULT_VARIABLE RETURNCODE
      OUTPUT_FILE "/dev/null"
      ERROR_FILE "/dev/null"
  )
  if(NOT "${RETURNCODE}" EQUAL 0)
    message(FATAL_ERROR "Python requirement ${REQUIREMENT} missing")
  endif()
endforeach()

# Check pyparsing version
execute_process(
    COMMAND bash -c "pip3 show pyparsing | grep 'Version: 2.4.7'"
    RESULT_VARIABLE RETURNCODE
    OUTPUT_FILE "/dev/null"
    ERROR_FILE "/dev/null"
)
if(NOT "${RETURNCODE}" EQUAL 0)
  message(FATAL_ERROR "pyparsing version 2.4.7 required. 3.x versions are known to cause build issues")
endif()
