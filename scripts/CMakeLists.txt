#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

# Check that required python dependencies are available
file(STRINGS "requirements.txt"
    REQUIREMENTS
    REGEX "^[^#].+$"
)
# pyparsing version 2.4.7 required. 3.x versions are known to cause build issues
list(APPEND REQUIREMENTS "pyparsing==2\\.4\\.7")

check_python_requirements(${REQUIREMENTS})
