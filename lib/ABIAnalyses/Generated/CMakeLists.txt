#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

# Code generation requirements

# Check if graph_tool is available
execute_process(COMMAND python3 -c "import pygraphviz"
  RESULT_VARIABLE HAS_PYGRAPHVIZ
  OUTPUT_VARIABLE PYGRAPHVIZ_OUTPUT
  ERROR_VARIABLE PYGRAPHVIZ_OUTPUT)

if(NOT HAS_PYGRAPHVIZ EQUAL "0")
  # Don't drop the whitespaces
  message(FATAL_ERROR "  
  Cannot find the pygraphviz module: ABIDataFlows.h cannot be generated.
  Please try one of the following commands:
  
  Debian/Ubuntu:
    apt-get install python3-pygraphviz
  
  Fedora/Red Hat/CentOS:
    yum install python3-pygraphviz
  
  pip (current user only):
    pip3 install --user pygraphviz
  
  In case of successful installation, the following command should succeed with no output:
    python3 -c 'import pygraphviz'")
endif()

# Generate headers for ABI analyses
set(ABIANALYSIS_TEMPLATE
  "${CMAKE_CURRENT_SOURCE_DIR}/ABIAnalysis.template")

set(ABIANALYSIS_GRAPHS
  "DeadRegisterArgumentsOfFunction"
  "DeadReturnValuesOfFunctionCall"
  "RegisterArgumentsOfFunctionCall"
  "UsedArgumentsOfFunction"
  "UsedReturnValuesOfFunctionCall"
  "UsedReturnValuesOfFunction")

foreach(GRAPH IN LISTS ABIANALYSIS_GRAPHS)
  message(STATUS "hey ${GRAPH}")
  add_custom_command(OUTPUT "${GRAPH}.h"
    COMMAND "${CMAKE_SOURCE_DIR}/scripts/monotone-framework-lattice.py"
      ${ABIANALYSIS_TEMPLATE} "${CMAKE_CURRENT_SOURCE_DIR}/${GRAPH}.dot" > "${GRAPH}.h"
    DEPENDS "${CMAKE_SOURCE_DIR}/scripts/monotone-framework-lattice.py"
            "${CMAKE_CURRENT_SOURCE_DIR}/${GRAPH}.dot"
            ${ABIANALYSIS_TEMPLATE}
    VERBATIM)
    list(APPEND ABIANALYSIS_HEADERS "${GRAPH}.h")
endforeach()

add_custom_target(abianalysisHeaders DEPENDS ${ABIANALYSIS_HEADERS})
