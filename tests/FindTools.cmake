#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

# Get the path to some system tools we'll need

set(LLC "${LLVM_TOOLS_BINARY_DIR}/llc")
find_program(DIFF diff)

# Check which architectures are supported, we need:
# * qemu-${ARCH}
# * A cross compiler (provided by the user)
# * libtinycode-${ARCH}.so, which must be in the search path

set(SUPPORTED_ARCHITECTURES "arm;i386;mips;mipsel;s390x;x86_64")
set(QEMU_BIN_PATH "${QEMU_INSTALL_PATH}/bin")
set(QEMU_LIB_PATH "${QEMU_INSTALL_PATH}/lib")

# We can test an architecture if we have a compiler and a libtinycode-*.so
foreach(ARCH ${SUPPORTED_ARCHITECTURES})
  find_library(LIBTINYCODE_${ARCH} "libtinycode-${ARCH}.so"
    HINTS "${QEMU_LIB_PATH}" "${CMAKE_INSTALL_PREFIX}/lib")
  find_program(QEMU_${ARCH} qemu-${ARCH} HINTS "${QEMU_BIN_PATH}")

  # If we miss one of the required components, drop the architecture
  if(NOT EXISTS "${LIBTINYCODE_${ARCH}}"
      OR NOT EXISTS "${C_COMPILER_${ARCH}}"
      OR NOT EXISTS "${QEMU_${ARCH}}")
    list(REMOVE_ITEM SUPPORTED_ARCHITECTURES ${ARCH})
  else()
    message("Testing enabled for ${ARCH}")
  endif()
endforeach()
