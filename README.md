# Purpose

`revng` is a static binary translator. Given a input ELF binary for one of the supported architectures (currently i386, x86-64, MIPS, ARM, AArch64 and s390x) it will analyze it and emit an equivalent LLVM IR. To do so, `revng` employs the QEMU intermediate representation (a series of TCG instructions) and then translates them to LLVM IR.

# How to build

`revng` employs CMake as a build system.
In order to build `revng`, use [orchestra](https://github.com/revng/orchestra) and make sure you're [building revng from source](https://github.com/revng/orchestra#building-from-source).

You can install as follows:

```sh
orc install revng
```

Remember to enter an `orc shell` to run `revng`:

```sh
orc shell
revng --help
```

You can run the test suite as follows:

```sh
orc install --test revng
```

# Example run

The simplest possible example consists in the following:

```sh
# Install revng
orc install revng

# Install the ARM toolchain
orc install toolchain/arm/gcc

# Enter in the build directory
orc shell -c revng

# Build
ninja

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
chmod +x hello.arm.translated

# Run translated version
./hello.arm.translated
# Hello, world!
```
