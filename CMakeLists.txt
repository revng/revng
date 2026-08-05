#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

cmake_minimum_required(VERSION 3.15.0)

project(revng)

include(share/revng/cmake/Common.cmake)

find_program(CLANG_PATH NAMES clang)

# Copy system configuration.yml
if(REVNG_SYSTEM_CONFIG AND EXISTS "${REVNG_SYSTEM_CONFIG}")
  set(REVNG_CONFIG_PATH "${CMAKE_BINARY_DIR}/etc/revng/configuration.yml")
  file(MAKE_DIRECTORY "${CMAKE_BINARY_DIR}/etc/revng")
  file(COPY_FILE "${REVNG_SYSTEM_CONFIG}" "${REVNG_CONFIG_PATH}")
  install(FILES "${REVNG_CONFIG_PATH}" DESTINATION etc/revng)
endif()

#
# Compile flags
#

# These have to be first to get highest priority
include_directories("${CMAKE_SOURCE_DIR}/include")
include_directories("${CMAKE_BINARY_DIR}/include")

add_definitions("-DINSTALL_PATH=\"${CMAKE_INSTALL_PREFIX}\"")

# Uncomment the following line if recursive coroutines make debugging hard
# add_definitions("-DDISABLE_RECURSIVE_COROUTINES")

# Remove -rdynamic
set(CMAKE_SHARED_LIBRARY_LINK_C_FLAGS)

# Basic compiler options
# ~~~
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=address -shared-libasan")
# ~~~
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++20")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wextra")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Werror")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-rtti")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-exceptions")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fvisibility-inlines-hidden")
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
# set(TUPLE_TREE_GENERATOR_EMIT_TRACKING_DEBUG ON)

if(${TUPLE_TREE_GENERATOR_EMIT_TRACKING_DEBUG})
  set(CMAKE_CXX_FLAGS
      "${CMAKE_CXX_FLAGS} -DTUPLE_TREE_GENERATOR_EMIT_TRACKING_DEBUG")
endif()

# Uncomment the following line if errors like `Couldn't find method
# SomeType::method` make debugging hard
#
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-limit-debug-info")

if(CMAKE_BUILD_TYPE STREQUAL "Release")
  add_flag_if_available("-Wno-unused-local-typedefs")
endif()

# Disable some warnings
add_flag_if_available("-Wno-unused-parameter")
add_flag_if_available("-Wno-unused-variable")
add_flag_if_available("-Wno-maybe-uninitialized")
add_flag_if_available("-Wno-init-list-lifetime")
add_flag_if_available("-Wno-ambiguous-reversed-operator")

# Add some extra warnings
add_flag_if_available("-Wstrict-aliasing")
add_flag_if_available("-fstrict-aliasing")

add_flag_if_available("-Wnon-virtual-dtor")
add_flag_if_available("-Wunreachable-code-break")
add_flag_if_available("-Winconsistent-missing-destructor-override")
add_flag_if_available("-Wnewline-eof")
add_flag_if_available("-Wmissing-prototypes")
add_flag_if_available("-Wimplicit-fallthrough")

add_definitions("-D_FILE_OFFSET_BITS=64")

check_cxx_compiler_flag("-no-pie" COMPILER_SUPPORTS_NO_PIE)

if(COMPILER_SUPPORTS_NO_PIE)
  set(NO_PIE "-no-pie")
endif()

include(CheckIncludeFiles)
check_include_files(valgrind/callgrind.h HAVE_VALGRIND_CALLGRIND_H)

if(HAVE_VALGRIND_CALLGRIND_H)
  add_definitions("-DHAVE_VALGRIND_CALLGRIND_H")
endif()

#
# Link LLVM
#
find_package(LLVM REQUIRED CONFIG)
include_directories(${LLVM_INCLUDE_DIRS})
add_definitions(${LLVM_DEFINITIONS})
link_directories(${LLVM_LIBRARY_DIRS})
llvm_map_components_to_libnames(
  LLVM_LIBRARIES
  core
  support
  irreader
  ScalarOpts
  linker
  Analysis
  object
  transformutils
  BitWriter
  BitReader
  InstCombine
  CodeGen
  Passes
  TargetParser)

# MLIR CMake stuff
find_package(MLIR REQUIRED CONFIG)
include_directories(${MLIR_INCLUDE_DIRS})

#
# Link Clang
#
find_package(Clang REQUIRED CONFIG)

#
# Component hash
#
file(MAKE_DIRECTORY "${CMAKE_BINARY_DIR}/share/revng/component-hashes")
set(COMPONENT_HASH_PATH
    "${CMAKE_BINARY_DIR}/share/revng/component-hashes/revng")
add_custom_command(
  OUTPUT "${COMPONENT_HASH_PATH}"
  COMMAND
    sh -c
    ' (git -C "${CMAKE_SOURCE_DIR}" rev-parse HEAD || echo "\"${CMAKE_PROJECT_VERSION}\"") '
    > "${COMPONENT_HASH_PATH}")
add_custom_target(generate-component-hash ALL DEPENDS "${COMPONENT_HASH_PATH}")
install(FILES "${COMPONENT_HASH_PATH}"
        DESTINATION "${CMAKE_INSTALL_DIR}/share/revng/component-hashes/")

#
# share/revng
#
add_custom_target(copy_share ALL COMMAND cp -Tar "${CMAKE_SOURCE_DIR}/share/"
                                         "${CMAKE_BINARY_DIR}/share/")

install(
  DIRECTORY "${CMAKE_BINARY_DIR}/share/"
  DESTINATION share/
  USE_SOURCE_PERMISSIONS)

# Export CMake targets
install(
  EXPORT revng
  NAMESPACE DESTINATION
  share/revng/cmake)

# Create share/revng/additional-bin-paths
set(ADDITIONAL_BIN_PATHS "${CMAKE_BINARY_DIR}/share/revng/additional-bin-paths")
file(WRITE "${ADDITIONAL_BIN_PATHS}" "libexec/revng\n")

file(RELATIVE_PATH RELATIVE_LLVM_TOOLS_BINARY_DIR "${CMAKE_INSTALL_PREFIX}"
     "${LLVM_TOOLS_BINARY_DIR}")
file(APPEND "${ADDITIONAL_BIN_PATHS}" "${RELATIVE_LLVM_TOOLS_BINARY_DIR}\n")

# Create additional-search-prefixes
file(WRITE "${CMAKE_BINARY_DIR}/additional-search-prefixes"
     "${CMAKE_INSTALL_PREFIX}\n")

file(RELATIVE_PATH RELATIVE_LLVM_TOOLS_BINARY_DIR "${CMAKE_INSTALL_PREFIX}"
     "${LLVM_TOOLS_BINARY_DIR}")
file(APPEND "${ADDITIONAL_BIN_PATHS}" "${RELATIVE_LLVM_TOOLS_BINARY_DIR}\n")

# Build the support module for each architecture and in several configurations
set(SUPPORT_MODULES_CONFIGS "normal;trace")
set(SUPPORT_MODULES_CONFIG_normal "")
set(SUPPORT_MODULES_CONFIG_trace "-DTRACE")

# The llvm modules whose functions can end up inlined in the decompiled code.
# Their names are the ones the C backend has to stay away from.
set(MODULES_TO_INLINE "")

foreach(
  ARCH
  aarch64
  arm
  mips
  mipsel
  x86_64
  i386
  s390x)
  set(OUTPUT "early-linked-${ARCH}.ll")
  add_custom_command(
    OUTPUT "${CMAKE_BINARY_DIR}/share/revng/${OUTPUT}"
    DEPENDS "${CMAKE_SOURCE_DIR}/share/revng/early-linked.c"
    COMMAND
      ${TARGET_CLANG} ARGS "${CMAKE_SOURCE_DIR}/share/revng/early-linked.c" -o
      "${CMAKE_BINARY_DIR}/share/revng/${OUTPUT}" -S -emit-llvm -g
      -DTARGET_${ARCH} -I"${CMAKE_SOURCE_DIR}/share/revng"
      -I"${CMAKE_CURRENT_SOURCE_DIR}/include")
  add_custom_target("early-linked-module-${OUTPUT}" ALL
                    DEPENDS "${CMAKE_BINARY_DIR}/share/revng/${OUTPUT}")
  add_dependencies(revng-all-binaries "early-linked-module-${OUTPUT}")

  # Enable the support for C exceptions to avoid optimizations that break
  # exception support when linking a module with isolated functions
  foreach(CONFIG ${SUPPORT_MODULES_CONFIGS})
    set(OUTPUT "support-${ARCH}-${CONFIG}.ll")
    add_custom_command(
      OUTPUT "${CMAKE_BINARY_DIR}/share/revng/${OUTPUT}"
      DEPENDS "${CMAKE_SOURCE_DIR}/share/revng/support.c"
      COMMAND
        ${TARGET_CLANG} ARGS "${CMAKE_SOURCE_DIR}/share/revng/support.c" -O2
        -fexceptions -o "${CMAKE_BINARY_DIR}/share/revng/${OUTPUT}" -S
        -emit-llvm -g -DTARGET_${ARCH} -I"${CMAKE_SOURCE_DIR}/share/revng"
        -I"${CMAKE_CURRENT_SOURCE_DIR}/include"
        ${SUPPORT_MODULES_CONFIG_${CONFIG}})
    add_custom_target("support-module-${OUTPUT}" ALL
                      DEPENDS "${CMAKE_BINARY_DIR}/share/revng/${OUTPUT}")
    add_dependencies(revng-all-binaries "support-module-${OUTPUT}")
  endforeach()

  # Annotate `libtcg-helpers-*.bc`. This loads the `libtcg-helpers` bitcode
  # files and runs analyses that determine which CSVs are accessed by each
  # helper function.
  set(LIBTCG_HELPERS
      "${CMAKE_INSTALL_PREFIX}/share/libtcg/libtcg-helpers-${ARCH}.bc")
  set(LIBTCG_HELPERS_FULL
      "${CMAKE_BINARY_DIR}/share/revng/libtcg-helpers-full-${ARCH}.bc")

  # Build-time helper-annotation pipeline. Order is significant:
  #
  # * `mark-inline-helpers-up-to` tags every helper that transitively reaches
  #   the configured runtime-library directories with the `revng_inline`
  #   section. Runs first so all subsequent passes see the tag set.
  # * `sroa` + `instsimplify` simplify the body of every helper so the
  #   downstream analyses see the smallest possible IR.
  # * `cpu-loop-exit` rewrites the CPU loop-exit pattern into something
  #   `analyze-helper-arguments` can understand.
  # * `analyze-helper-arguments` discovers, for every helper, which arguments
  #   alias CSVs at the call site.
  # * `loop-unroll` (x86 only, see below) unrolls loops so the subsequent
  #   `fix-helpers` pass can rewrite the accesses contained in bounded loop
  #   bodies.
  # * `fix-helpers` rewrites helper bodies to access CSVs via the env pointer,
  #   switching on env-relative offsets.
  # * `detect-uninlinable-helpers` excludes not viable helpers from inlining.

  # Only the x86 softfloat helpers iterate over the vector-register lanes with
  # bounded `for` loops (e.g. the packed `helper_*p[sd]` operate on
  # `XMM/ZMM_D(i)`). `fix-helpers` can only rewrite the per-lane CSV accesses
  # once those loops are unrolled. This step is not needed on other archs, so we
  # avoid it to save space on the generated libtcg helpers and build time.
  set(EXTRA_HELPER_PASSES "")
  if(ARCH STREQUAL "x86_64" OR ARCH STREQUAL "i386")
    set(EXTRA_HELPER_PASSES -loop-unroll)
  endif()

  add_custom_command(
    OUTPUT "${LIBTCG_HELPERS_FULL}"
    DEPENDS "${LIBTCG_HELPERS}" revng-all-binaries revngHelperArgumentsAnalysis
            revngHelperInliningAnalyses
    COMMAND
      ./bin/revng opt -mark-inline-helpers-up-to -sroa -instsimplify
      -cpu-loop-exit -analyze-helper-arguments ${EXTRA_HELPER_PASSES}
      -fix-helpers -fix-helpers-architecture="${ARCH}" -dce
      -detect-uninlinable-helpers "${LIBTCG_HELPERS}" -o
      "${LIBTCG_HELPERS_FULL}")
  add_custom_target("annotated-libtcg-helpers-${ARCH}" ALL
                    DEPENDS "${LIBTCG_HELPERS_FULL}")

  function(process_helper_module)
    set(options)
    set(oneValueArgs NAME)
    set(multiValueArgs PASSES)
    cmake_parse_arguments(OPT "${options}" "${oneValueArgs}"
                          "${multiValueArgs}" ${ARGN})

    set(PROCESS_HELPER_MODULE_OUTPUT
        "${CMAKE_BINARY_DIR}/share/revng/libtcg-helpers-${OPT_NAME}-${ARCH}.bc")
    list(TRANSFORM OPT_PASSES PREPEND -)
    add_custom_command(
      OUTPUT "${PROCESS_HELPER_MODULE_OUTPUT}"
      DEPENDS "${LIBTCG_HELPERS_FULL}" revng-all-binaries
              revngHelperArgumentsAnalysis revngHelperInliningAnalyses
      COMMAND ./bin/revng opt ${OPT_PASSES} "${LIBTCG_HELPERS_FULL}" -o
              "${PROCESS_HELPER_MODULE_OUTPUT}")
    add_custom_target("annotated-libtcg-helpers-${OPT_NAME}-${ARCH}" ALL
                      DEPENDS "${PROCESS_HELPER_MODULE_OUTPUT}")
  endfunction()

  # Produce to-inline version of helpers. In this version the bitcode file only
  # contains helper functions that are in the `revng_inline` section and the
  # CSVs that are used by them. This is used by `inline-helpers` to allow
  # inlining the body of the `revng_inline` function into the isolated
  # functions' bodies.
  process_helper_module(
    NAME
    to-inline
    PASSES
    slim-down-helpers-module
    sroa
    instsimplify
    simplifycfg
    strip-dead-debug-info)

  list(APPEND MODULES_TO_INLINE
       "${CMAKE_BINARY_DIR}/share/revng/libtcg-helpers-to-inline-${ARCH}.bc")

  # Produce declarations-only version of helpers. In this version of the bitcode
  # file all the helper functions have been turned into declarations and all
  # CSVs have been dropped. This will be used in `lift` to import the helper
  # declarations to be used in the lifting process.
  process_helper_module(
    NAME declarations-only PASSES helpers-module-to-declarations globaldce
    strip-dead-debug-info)
endforeach()

# Custom command to create .clang-format file from revng-check-conventions
add_custom_command(
  OUTPUT "${CMAKE_BINARY_DIR}/share/revng/.clang-format"
  DEPENDS "${CMAKE_SOURCE_DIR}/libexec/revng/check-conventions"
          "${CMAKE_SOURCE_DIR}/share/revng/clang-format-style-file.yml"
          copy_share copy_libexec
  COMMAND
    "${CMAKE_BINARY_DIR}/libexec/revng/check-conventions" ARGS
    --print-clang-format-config >
    "${CMAKE_BINARY_DIR}/share/revng/.clang-format")

add_custom_target(clang-format-dot-file ALL
                  DEPENDS "${CMAKE_BINARY_DIR}/share/revng/.clang-format")

#
# libexec/revng
#
add_custom_target(
  copy_libexec ALL COMMAND cp -Tar "${CMAKE_SOURCE_DIR}/libexec/"
                           "${CMAKE_BINARY_DIR}/libexec/")
install(
  DIRECTORY "${CMAKE_BINARY_DIR}/libexec/"
  DESTINATION libexec/
  USE_SOURCE_PERMISSIONS)

#
# Export information useful for subdirectories
#
set(PYTHON_GENERATED_MODEL_PATH revng/model/_generated.py)

#
# Enable CTest
#
enable_testing()

# ~~~
# Find Python using the FindPython module. We request two components:
# * Interpreter: needed for some variables, such as $Python_SITELIB
# * Development: needed to compile nanobind module and friends, provides the
#                $Python_INCLUDE_DIRS and $Python_LIBRARIES variables
# ~~~
find_package(
  Python3
  COMPONENTS Interpreter Development
  REQUIRED)
file(RELATIVE_PATH PYTHON_INSTALL_PATH "${CMAKE_INSTALL_PREFIX}"
     "${Python3_SITELIB}")

#
# Build docs
#
configure_file(mkdocs.yml "${CMAKE_BINARY_DIR}" COPYONLY)
set(HTML_DOCS_PATH "${CMAKE_CURRENT_BINARY_DIR}/share/doc/revng/html")
find_program(MKDOCS NAMES mkdocs)

add_custom_command(
  OUTPUT share/doc/revng/html/404.html
  DEPENDS
    mkdocs.yml
    copy_share
    "share/doc/revng/references/abi-definition.md"
    "share/doc/revng/references/mime-types.md"
    "share/doc/revng/references/ptml.md"
    "share/doc/revng/references/cli/revng-project-analyze.md"
    "share/doc/revng/references/cli/revng-project-artifact.md"
    "share/doc/revng/user-manual/index.md"
    "share/doc/revng/user-manual/initial-setup.md"
    "share/doc/revng/user-manual/key-concepts/metaaddress.md"
    "share/doc/revng/user-manual/key-concepts/model.md"
    "share/doc/revng/user-manual/key-concepts/artifacts-and-analyses.md"
    "share/doc/revng/user-manual/tutorial/model-from-scratch.md"
    "share/doc/revng/user-manual/tutorial/running-initial-auto-analysis.md"
    "share/doc/revng/user-manual/tutorial/comments-and-local-variables.md"
    "share/doc/revng/what-is-revng.md"
    "share/doc/revng/index.md"
    generate-revngModel-tuple-tree-code
    python/revng/internal/cli/_commands/process_docs_yaml/pipeline.md.tpl
    python/revng/internal/cli/_commands/process_docs_yaml/artifacts.md.tpl
    python/revng/internal/cli/_commands/process_docs_yaml/analyses.md.tpl
    python/revng/internal/cli/_commands/process_docs_yaml/common.md.tpl
    python/revng/internal/pipeline.yml
  WORKING_DIRECTORY "${CMAKE_BINARY_DIR}"
  COMMAND
    "./bin/revng" process-docs-yaml
    "${CMAKE_SOURCE_DIR}/python/revng/internal/cli/_commands/process_docs_yaml/pipeline.md.tpl"
    < "${CMAKE_SOURCE_DIR}/python/revng/internal/pipeline.yml" >
    share/doc/revng/references/pipeline.md
  COMMAND
    "./bin/revng" process-docs-yaml
    "${CMAKE_SOURCE_DIR}/python/revng/internal/cli/_commands/process_docs_yaml/analyses.md.tpl"
    < "${CMAKE_SOURCE_DIR}/python/revng/internal/pipeline.yml" >
    share/doc/revng/references/analyses.md
  COMMAND
    "./bin/revng" process-docs-yaml
    "${CMAKE_SOURCE_DIR}/python/revng/internal/cli/_commands/process_docs_yaml/artifacts.md.tpl"
    < "${CMAKE_SOURCE_DIR}/python/revng/internal/pipeline.yml" >
    share/doc/revng/references/artifacts.md
  COMMAND env NO_MKDOCS_2_WARNING=1 ${MKDOCS} build --quiet --strict --site-dir
          mkdocs-output
  COMMAND rm -rf "${HTML_DOCS_PATH}"
  COMMAND mv -T mkdocs-output "${HTML_DOCS_PATH}"
  COMMENT "Generating mkdocs")
add_custom_target(mkdocs ALL DEPENDS share/doc/revng/html/404.html)

#
# Gather helper names
#
add_custom_command(
  OUTPUT "${CMAKE_BINARY_DIR}/share/revng/helper-list.csv"
  COMMAND
    ${CMAKE_SOURCE_DIR}/scripts/extract-helper-names.sh ${MODULES_TO_INLINE} >
    "${CMAKE_BINARY_DIR}/share/revng/helper-list.csv"
  DEPENDS copy_share ${MODULES_TO_INLINE})
add_custom_target(helper-list ALL
                  DEPENDS "${CMAKE_BINARY_DIR}/share/revng/helper-list.csv")

#
# Proceed to subdirectories
#
add_subdirectory(docs)
add_subdirectory(include)
add_subdirectory(lib)
add_subdirectory(python)
add_subdirectory(tests)
add_subdirectory(tools)
add_subdirectory(typescript)
