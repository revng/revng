.. warning::
    This document is significantly outdated and is preserved for historical
    purpose until a more modern version of this content is added to the official
    documentation.
    For the latest documentation go to `docs.rev.ng <https://docs.rev.ng/>`_.

***************
rev.ng Overview
***************

The goal of this document is to briefly introduce key concepts about the design of rev.ng.

rev.ng is a framework designed to produce a high-level and self-contained representation of what a binary program does.
This process is also known as *lifting* a binary.

This high-level representation of choice is the LLVM IR.

Once rev.ng has produced the LLVM IR, there are two core things that you can do: analyze it or recompile it.

In the former case, the goal is to understand what the program does.
One of the core goals of the rev.ng project as a whole is, in fact, to build a decompiler, i.e., a tool able to produce C code representing the behavior of the program.

The other use case for rev.ng is to recompile the lifted IR to an executable binary.
Typically, we do this for ensuring the accuracy of the lifting process, but there are interesting use cases.
For example, a user might want to add off-the-shelf LLVM instrumentations such as `SanitizerCoverage <https://clang.llvm.org/docs/SanitizerCoverage.html>`_ and `AddressSanitizer <https://clang.llvm.org/docs/AddressSanitizer.html>`_ in order to perform coverage-guided fuzzing on binaries using `libFuzzer <https://llvm.org/docs/LibFuzzer.html>`_.
If this sounds interesting for you, check out the `dedicated blog post <https://rev.ng/blog/fuzzing/post.html>`_.

Recompiling LLVM IR to executable code is trivial.
After all, LLVM is a compiler framework.

The Lifting Process
-------------------

The lifting process requires us to be able to represent the semantic of each instruction of the ISA of the input binary.
This is a very large endeavor and it's highly error prone.

To support as many input architectures as possible with a reduced effort, we exploit an existing open source piece of software able to represent the behavior of each instruction of many different CPUs: QEMU.

A QEMU Primer
~~~~~~~~~~~~~

QEMU is a well-known emulator.
The peculiarity of QEMU is that it does not *interpret* each input instruction one after the other, but it translates them in an equivalent piece of code for the target architecture.

For instance, assuming an ARM program being emulated on an x86-64 host, the ``mov r0, r1`` instruction will be translated in a series of equivalent x86-64 instructions acting on a ``struct`` representing the CPU state of the emulated ARM CPU.

Emulators producing code for the host architecture at run-time, such as QEMU, are known as *dynamic binary translators*.

What's even more interesting about QEMU is that its internal design is quite similar to a compiler.
Indeed, the translation process from the input architecture A to the host architecture B uses an intermediate representation that it's similar in many aspects to LLVM IR.
This intermediate representation is known as *tiny code instructions*::

    ARM  >----         ----> s390x
              \       /
    x86  >---  \     /  ---> AArch64
             \  \   /  /
             tiny code
             /  /   \  \
    MIPS >---  /     \  ---> x86
              /       \
    ...  >----         ----> ...

Just as a compiler, QEMU is divided into:

* a frontend: transforms each input instruction into tiny code instructions;
* a mid-end: performs a set of target-independent, lightweight optimizations such as constant propagation and dead code elimination;
* a backend: takes care of producing code that can be executed by the host CPU;

As of today, QEMU features frontends for x86, x86-64, ARM, AArch64, MIPS, SPARC, s390x, RISCV, PowerPC, Hexagon, and many others

In terms of backend, on the other hand, the following architectures are supported: x86, x86-64, ARM, MIPS, AArch64, s390, RISCV, PowerPC, and SPARC.
QEMU also features a tiny code interpreter, ``tci``, that enables it to run even on target architectures for which a backend is not available.

Now that we have a basic understanding of how QEMU works, let's take a look at some actual tiny code instructions.

.. code-block::

    $ qemu-x86_64 -d in_asm,op /bin/bash
    IN:
    0x00000040008e2050:  mov    rdi,rsp

    OP:
     ld_i32 tmp11,env,$0xfffffffffffffffc
     movi_i32 tmp12,$0x0
     brcond_i32 tmp11,tmp12,ne,$L0

     ---- 0x40008e2050
     mov_i64 tmp0,rsp
     mov_i64 rdi,tmp0
     movi_i64 tmp3,$0x40008e2053
     st_i64 tmp3,env,$0x80
     exit_tb $0x0
     set_label $L0
     exit_tb $0x7ffff3072013

Here we're emulating ``bash`` of the host system using ``qemu-x86_64``, QEMU's Linux user space emulator.
The ``-d`` flag enables debug output: ``in_asm`` shows the input instruction currently being translated, while ``op`` dumps the produced tiny code instructions.

Unlike instructions of actual CPUs, the tiny code instructions do just one thing and have no side effects.

In rev.ng, we want to be able to employ QEMU frontends to produce tiny code instructions from a set of raw bytes.
Working on tiny code instructions enables us to handle any architecture supported by QEMU in a unified way.

In order to use QEMU as a library, we maintain a `fork of QEMU <https://github.com/revng/qemu>`_ that exposes the various frontends through a dynamic library known as ``libtinycode`` (e.g., ``libtinycode-arm.so``).

Helper Functions
~~~~~~~~~~~~~~~~

Not all of the instructions are handled completely through tiny code.
As an example, consider the floating point division instruction.
QEMU, to handle all the subtle differences between floating point implementations in the various CPU it supports, implements floating point instructions in software (i.e., it does not use the host processor's floating point instructions).

Implementing the semantics of a floating point division in a tiny code frontend would be very hard and bring little to no benefit.
Therefore, tiny code features a special type of instruction that enables the developer to call an external function written in C that can directly manipulate the CPU state.

Such functions are known as *helper functions*.

To produce a self-contained representation of the input program, in rev.ng, we need helper functions too.
Therefore, in our fork of QEMU, we do not just produce libtinycode, but also a set of files containing the LLVM IR for the helper functions targeting each of the frontends.
Producing files like those is rather easy since they are written in C and clang can easily produce LLVM IR from them.

The helper functions in LLVM IR form are collected in files such as ``libtinycode-helpers-arm.bc``, which are then installed and made available to rev.ng.

The Code Discovery Process
~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``revng-lift`` program takes care of the lifting process.

One of the main things ``revng-lift`` does is identifying all the executable code present in the binary.
More specifically, it identifies as many *jump targets* as possible.
A jump target can be seen as the start address of a basic block.

Briefly, here's what it does:

#. Load the program data and executable code.
#. Scan global data (e.g., ``.rodata``) for pointer-size integers that have values that seem to point to executable code.
#. Scan the binary's metadata for entry points (e.g., program entry point and exported functions).
#. Initialize a list of *jump targets* to inspect with the previously collected entry points/pointers to code. For each jump target:

   #. Use ``libtinycode`` to obtain tiny code instruction for the code at the currently considered address.
   #. Translate each tiny code instruction into equivalent LLVM IR.
   #. Register the target of direct branch instructions to be visited.
   #. Once all the direct branch instructions have been translated, perform more aggressive analyses to detect all the possible targets of indirect jump instructions (e.g., ``jmp rax``).
      These instructions are typically generated by indirect function calls and ``switch`` statements in C.
   #. Repeat.

#. Finalize the module and emit it.

Converting tiny code instructions into LLVM IR is of key importance.
In fact, the QEMU IR (tiny code instructions) is designed to be optimized at run-time, therefore it's not suitable to perform sophisticated analyses.
On the other hand, the LLVM is a full-fledged compiler framework where it is possible and it makes sense to perform aggressive analyses and transformations.

To know more about what the LLVM IR we produce looks like, proceed to `GeneratedIRReference.rst <GeneratedIRReference.rst>`_.
