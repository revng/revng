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
  set(C_COMPILER_${ARCH} "")

  find_library(LIBTINYCODE_${ARCH} "libtinycode-${ARCH}.so"
    HINTS "${QEMU_LIB_PATH}" "${CMAKE_INSTALL_PREFIX}/lib")
  find_program(QEMU_${ARCH} qemu-${ARCH} HINTS "${QEMU_BIN_PATH}")

  # Try to to autodetect the compiler looking for arch*-(musl|uclibc)*-gcc in
  # PATH
  string(REPLACE ":" ";" PATH "$ENV{PATH}")
  foreach(SEARCH_PATH IN LISTS PATH)
    if (NOT C_COMPILER_${ARCH})
      set(MUSL_TOOLCHAIN "")
      set(UCLIBC_TOOLCHAIN "")
      set(TOOLCHAIN "")

      file(GLOB MUSL_TOOLCHAIN "${SEARCH_PATH}/${ARCH}*-musl*-gcc")
      file(GLOB UCLIBC_TOOLCHAIN "${SEARCH_PATH}/${ARCH}*-uclibc*-gcc")
      if(MUSL_TOOLCHAIN)
        set(TOOLCHAIN "${MUSL_TOOLCHAIN}")
      endif()
      if(UCLIBC_TOOLCHAIN)
        set(TOOLCHAIN "${UCLIBC_TOOLCHAIN}")
      endif()

      if(TOOLCHAIN)
        set(C_COMPILER_${ARCH} "${TOOLCHAIN}")
        message("${ARCH} compiler autodetected: ${C_COMPILER_${ARCH}}")
      endif()

    endif()
  endforeach()

  # If we miss one of the required components, drop the architecture
  if(LIBTINYCODE_${ARCH} STREQUAL "LIBTINYCODE_${ARCH}-NOTFOUND"
      OR C_COMPILER_${ARCH} STREQUAL "C_COMPILER_${ARCH}-NOTFOUND"
      OR QEMU_${ARCH} STREQUAL "QEMU_${ARCH}-NOTFOUND")
    list(REMOVE_ITEM SUPPORTED_ARCHITECTURES ${ARCH})
  else()
    message("Testing enabled for ${ARCH}")
  endif()
endforeach()
