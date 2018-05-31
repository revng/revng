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

set(SUPPORTED_ARCHITECTURES "aarch64;alpha;arm;armeb;cris;i386;m68k;microblaze;microblazeel;mips;mips64;mips64el;mipsel;mipsn32;mipsn32el;nbd;or32;ppc;ppc64;ppc64abi32;s390x;sh4;sh4eb;sparc;sparc32plus;sparc64;unicore32;x86_64")
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
