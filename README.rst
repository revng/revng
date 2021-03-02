*******
Purpose
*******

``revng`` is a static binary translator. Given a input ELF binary for one of the
supported architectures (currently i386, x86-64, MIPS, ARM, AArch64 and s390x)
it will analyze it and emit an equivalent LLVM IR. To do so, ``revng`` employs
the QEMU intermediate representation (a series of TCG instructions) and then
translates them to LLVM IR.

************
How to build
************

``revng`` employs CMake as a build system.
In order to build ``revng``, use orchestra:

    https://github.com/revng/orchestra

To run the test suite simply, from the build directory, run:

.. code-block:: sh

    # Enter in the build directory
    orc shell -c revng

    # Run the tests
    ctest -j$(nproc)

***********
Example run
***********

The simplest possible example consists in the following:

.. code-block:: sh

    # Install the ARM toolchain
    orc install toolchain/arm/gcc

    # Enter in the build directory
    orc shell -c revng

    # Build programs (skip building test material)
    ninja revng-all-binaries

    # Create hello world program
    cat > hello.c <<EOF
    #include <stdio.h>

    int main(int argc, char *argv[]) {
      printf("Hello, world!\n");
    }
    EOF

    # Compile
    armv7a-hardfloat-linux-uclibceabi-gcc \
      -Wl,-Ttext-segment=0x20000 \
      -static hello.c \
      -o hello.arm

    # Translate
    ./bin/revng translate hello.arm

    # Run translated version
    ./hello.arm.translated
    # Hello, world!
