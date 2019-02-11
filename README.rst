*******
Purpose
*******

`revng` is a static binary translator. Given a input ELF binary for one of the
supported architectures (currently MIPS, ARM and x86-64) it will analyze it and
emit an equivalent LLVM IR. To do so, `revng` employs the QEMU intermediate
representation (a series of TCG instructions) and then translates them to LLVM
IR.

************
How to build
************

`revng` employs CMake as a build system. The build system will try to
automatically detect the QEMU installation and the GCC toolchains require to
build the test binaries.

If everything is in standard locations, you can just run::

    mkdir build/
    cd build/
    cmake ..
    make -j$(nproc)
    make install

For further build options and more advanced configurations see
docs/BuildSystem.rst (TODO: reference).

To run the test suite simply run::

    make test

***********
Example run
***********

The simplest possible example consists in the following::

    cd build
    cat > hello.c <<EOF
    #include <stdio.h>

    int main(int argc, char *argv[]) {
      printf("Hello, world!\n");
    }
    EOF
    armv7a-hardfloat-linux-uclibceabi-gcc -static hello.c -o hello.arm
    ./translate hello.arm
    # ...
    ./hello.arm.translated
    Hello, world!
