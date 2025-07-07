execute_process(
  COMMAND "${Python_EXECUTABLE}" "-c"
          "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))"
  OUTPUT_VARIABLE PYTHON_EXT_SUFFIX
  OUTPUT_STRIP_TRAILING_WHITESPACE)

llvm_map_components_to_libnames(LLVM_LIBRARIES Support Core)

add_library(my_ext MODULE my_ext.cpp)
set_target_properties(
  my_ext
  PROPERTIES LINKER_LANGUAGE CXX
             PREFIX ""
             SUFFIX "${PYTHON_EXT_SUFFIX}")
target_link_libraries(my_ext PUBLIC ${Python_LIBRARIES} ${LLVM_LIBRARIES}
                                    nanobind revngRegistry revngSupport)
target_include_directories(my_ext PUBLIC ${Python_INCLUDE_DIRS})

install(
  FILES "${CMAKE_CURRENT_BINARY_DIR}/my_ext${PYTHON_EXT_SUFFIX}"
  DESTINATION "${CMAKE_INSTALL_PREFIX}/${PYTHON_INSTALL_PATH}/revng/internal")

revng_add_library_internal(fakePipebox SHARED fake_pipebox.cpp)
target_link_libraries(
  fakePipebox
  ${Python_LIBRARIES}
  ${LLVM_LIBRARIES}
  nanobind
  revngRegistry
  revngSupport
  revngModel)
target_include_directories(fakePipebox PUBLIC ${Python_INCLUDE_DIRS})
