#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

set(MASS_TESTING_REPORT_FILES
    package.json
    src/binary.html
    src/crashes.html
    src/failures.html
    src/index.html
    src/main.css
    src/main.ts
    src/ooms.html
    src/raw_data.html
    src/successes.html
    src/timeouts.html
    tsconfig.json
    webpack.config.js)

foreach(INPUT_FILE ${MASS_TESTING_REPORT_FILES})
  configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/${INPUT_FILE}"
    "${CMAKE_BINARY_DIR}/mass-testing-report/${INPUT_FILE}" COPYONLY)
endforeach()

add_custom_command(
  OUTPUT
    "${CMAKE_BINARY_DIR}/mass-testing-report/node_modules/.package-lock.json"
  COMMAND npm install --silent
  DEPENDS "${CMAKE_BINARY_DIR}/mass-testing-report/package.json"
  WORKING_DIRECTORY "${CMAKE_BINARY_DIR}/mass-testing-report")

add_custom_target(
  npm-install-mass-testing-report
  DEPENDS
    "${CMAKE_BINARY_DIR}/mass-testing-report/node_modules/.package-lock.json"
    "${CMAKE_BINARY_DIR}/mass-testing-report/package.json")

add_custom_command(
  OUTPUT "${CMAKE_BINARY_DIR}/mass-testing-report/dist/main.js"
         "${CMAKE_BINARY_DIR}/mass-testing-report/dist/crashes.html"
         "${CMAKE_BINARY_DIR}/mass-testing-report/dist/failures.html"
         "${CMAKE_BINARY_DIR}/mass-testing-report/dist/index.html"
         "${CMAKE_BINARY_DIR}/mass-testing-report/dist/ooms.html"
         "${CMAKE_BINARY_DIR}/mass-testing-report/dist/raw_data.html"
         "${CMAKE_BINARY_DIR}/mass-testing-report/dist/timeouts.html"
  COMMAND rm -rf dist && npm run build > /dev/null 2>&1
  DEPENDS
    "${CMAKE_BINARY_DIR}/mass-testing-report/node_modules/.package-lock.json"
  WORKING_DIRECTORY "${CMAKE_BINARY_DIR}/mass-testing-report")

add_custom_target(
  mass-testing-report-generate ALL
  DEPENDS "${CMAKE_BINARY_DIR}/mass-testing-report/dist/main.js")

install(DIRECTORY "${CMAKE_BINARY_DIR}/mass-testing-report/dist/"
        DESTINATION "${CMAKE_INSTALL_PREFIX}/share/mass-testing-report/")
