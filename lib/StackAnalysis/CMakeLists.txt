#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

# Check if graph_tool is available
execute_process(COMMAND python -c "import pygraphviz"
  RESULT_VARIABLE HAS_PYGRAPHVIZ
  OUTPUT_VARIABLE PYGRAPHVIZ_OUTPUT
  ERROR_VARIABLE PYGRAPHVIZ_OUTPUT)

if(NOT HAS_PYGRAPHVIZ EQUAL "0")
  # Don't drop the whitespaces
  message(FATAL_ERROR "  
  Cannot find the pygraphviz module: abidataflows.h cannot be generated.
  Please try one of the following commands:
  
  Debian/Ubuntu:
    apt-get install python-pygraphviz
  
  Fedora/Red Hat/CentOS:
    yum install python2-pygraphviz
    yum install python3-pygraphviz
  
  pip (current user only):
    pip install --user pygraphviz
  
  In case of successful installation, the following command should succeed with no output:
    python -c 'import pygraphviz'")
endif()

# Generate classes for the ABI data flow analyses
set(ABIDATAFLOWS_SOURCES
  "${CMAKE_CURRENT_SOURCE_DIR}/abidataflows-header.h"
  "${CMAKE_CURRENT_SOURCE_DIR}/abidataflows-footer.h"
  "${CMAKE_CURRENT_SOURCE_DIR}/dead-register-arguments-of-function.dot"
  "${CMAKE_CURRENT_SOURCE_DIR}/dead-return-values-of-function-call.dot"
  "${CMAKE_CURRENT_SOURCE_DIR}/register-arguments-of-function-call.dot"
  "${CMAKE_CURRENT_SOURCE_DIR}/used-arguments-of-function.dot"
  "${CMAKE_CURRENT_SOURCE_DIR}/used-return-values-of-function-call.dot"
  "${CMAKE_CURRENT_SOURCE_DIR}/used-return-values-of-function.dot")
add_custom_command(OUTPUT abidataflows.h
  COMMAND "${CMAKE_SOURCE_DIR}/scripts/monotone-framework.py"
    --call-arcs ${ABIDATAFLOWS_SOURCES} > abidataflows.h
  DEPENDS "${CMAKE_SOURCE_DIR}/scripts/monotone-framework.py"
           ${ABIDATAFLOWS_SOURCES}
  VERBATIM)
add_custom_target(abidataflows DEPENDS abidataflows.h)

add_library(StackAnalysis STATIC
  stackanalysis.cpp functionssummary.cpp intraprocedural.cpp interprocedural.cpp
  element.cpp functionabi.cpp cache.cpp abiir.cpp incoherentcallsanalysis.cpp
  monotoneframeworkexample.cpp)
target_include_directories(StackAnalysis
  PRIVATE
  "${CMAKE_CURRENT_BINARY_DIR}")
add_dependencies(StackAnalysis abidataflows)
