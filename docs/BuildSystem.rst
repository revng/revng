***********************************
Identifying the required components
***********************************

`revng` requires three main components: LLVM, QEMU and one or more GCC
toolchains.

LLVM
====

LLVM should be automatically detected if it's a standard location. If you want
to use a custom build in a non-standard location use the ``LLVM_DIR`` option,
making it point to the directory containing ``LLVMConfig.cmake`` in the install
directory (typically ``$INSTALL_PATH/share/llvm/cmake``):

.. code-block:: sh

    cmake -DLLVM_DIR=/home/me/llvm-install/share/llvm/cmake ../

QEMU
====

By default the build system will look for the system QEMU installation (i.e., in
``/usr/``). If this is not what you want you can customize the QEMU install path
using the ``QEMU_INSTALL_PATH`` variable:

.. code-block:: sh

    cmake -DQEMU_INSTALL_PATH="/home/me/qemu-install/" ../

GCC
===

The `revng` build system will try to automatically detect toolchains to compile
code for the supported architectures. The toolchains are required to correctly
run the `revng` test suite.

The autodetction mechanism looks in all the directories in the ``PATH``
environment variable for an executable matching the pattern ``$ARCH*-musl*-gcc``
or ``$ARCH*-uclibc*-gcc`` (glibc is currently unsupported).

It's possible to force the usage of a specific compiler for a certain
architecture using the CMake variable ``C_COMPILER_$ARCH``:

.. code-block:: sh

    cmake -DC_COMPILER_mips="/home/me/my-mips-compiler" \
          -DC_COMPILER_miparm="/home/me/my-arm-compiler" \
          -DC_COMPILER_x86_64="/home/me/my-x86_64-compiler" \
          ../

The `revng` build system also provides an option to specify additional flags to
employ when using the above mentioned commpilers for compilation and linking of
test binaries. This can be done using the variables ``TEST_CFLAGS_$ARCH`` (for
compile-time flags) and ``TEST_LINK_LIBRARIES_$ARCH`` (for linking).

This is particularly useful to force x86-64 binaries to compile with software
floating point, which can be achieved with GCC as follows:

.. code-block:: sh

    cmake -DTEST_CFLAGS_x86_64="-msoft-float -mfpmath=387 -mlong-double-64" \
          -DTEST_LINK_LIBRARIES_x86_64="-lc $LLVM_INSTALL_PATH/lib/linux/libclang_rt.builtins-x86_64.a" \
          ../

********************
Common CMake options
********************

In the following we provide some useful-to-know CMake variables, that are not
specific to `revng`. They can be specified using the ``-D`` flag, e.g.:

.. code-block:: sh

    cmake -DCMAKE_INSTALL_PREFIX=/tmp ../

:CMAKE_INSTALL_PREFIX: Specify the install path. This is the destination
                       directory where all the necessary files will be copied
                       upon ``make install``.
:CMAKE_BUILD_TYPE: Specify the type of build. The two most useful values are
                   ``Debug``, for an unoptimized build with debugging
                   information, and ``RelWithdebinfo``, for an optimized build
                   with debugging information.
