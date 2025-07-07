add_library(fakePipebox SHARED fake_pipebox.cpp)

llvm_map_components_to_libnames(LLVM_LIBRARIES Support Core)
target_link_libraries(
  fakePipebox
  ${Python_LIBRARIES}
  ${LLVM_LIBRARIES}
  nanobind
  revngPypeline
  revngSupport
  revngModel)

target_include_directories(fakePipebox PUBLIC ${Python_INCLUDE_DIRS})

revng_add_test(NAME pypeline-test COMMAND "${CMAKE_CURRENT_SOURCE_DIR}/test.py"
               "${CMAKE_CURRENT_BINARY_DIR}/libfakePipebox.so")
set_tests_properties(
  pypeline-test
  PROPERTIES LABELS "pypeline;test" ENVIRONMENT
             "PYTHONPATH=${CMAKE_BINARY_DIR}/${PYTHON_INSTALL_PATH}")
