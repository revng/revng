***************************************************
Using `revng` with Python: a simple instrumenation
***************************************************

In this document we'll guide the user through using `revng`'s output from
Python. Among the many possibilities that arouse from the LLVM IR provided by
`revng`, in this document we'll show how it's possible to perform a simple
instrumentation of an existing binary, by injecting some code in the generated
LLVM IR and recompiling it.

To manipulate LLVM IR from Python we'll use `llvmcpy`_, a lightweight wrapper
around the LLVM-C API for Python.

An Hello World for ARM
======================

Let's first of all create a simple program in C called `hello.c`:

.. code-block:: c

    #include <stdio.h>

    int main(int argc, char *argv[]) {
      printf("Hello world!\n");
      return 0;
    }

We can compile `hello.c` for ARM and link it statically:

.. code-block:: sh

    armv7a-hardfloat-linux-uclibceabi-gcc hello.c -o hello -static

Using the `translate`_ tool we can have `revng` produce the
LLVM IR and recompile it for us. The output should be a working
``hello.translated`` program for x86-64 (our host architecture):

.. code-block:: sh

    $ translate hello
    $ ./hello.translated
    Hello world!

All good!

Tracing all the syscall invocations
===================================

For this example, we'll write a simple Python script (``instrument.py``) which
takes in input the `revng` generated LLVM IR, identifies all the syscalls and
instrument them injecting the code to print the number of syscall to be
performed.

In the ARM architecture, syscalls are expressed as calls to a function whose
name begins with `helper_exception_with_syndrome_`. After this prefix there is
an index identifying the specialization of the function, as explained in
`GeneratedIRReference.rst`_, but this is not interesting for us in this case.

Once we identified all the calls to the said functions, we will simply load the
value of the `r7` register, which holds the number of the syscall, and print it
to `stderr` using the `dprintf` function.

First of all we need to import `llvmcpy`_,obtain the default `LLVMContext`
object and load the input LLVM IR:

.. code-block:: python

    from llvmcpy import llvm
    context = llvm.get_global_context()
    buffer = llvm.create_memory_buffer_with_contents_of_file(sys.argv[1])
    module = context.parse_ir(buffer)

Now that we a reference to the module produced by `revng` we can collect the
objects required to perform the `dprintf` call, i.e., the function itself, the
CSV representing the register `r7`, a constant integer representing `stderr` and
the format string for `dprintf`:

.. code-block:: python

    r7 = module.get_named_global("r7")

    dprintf = module.get_named_function("dprintf")

    two = context.int32_type().const_int(2, True)

    message_str = context.const_string("%d\n", 4, True)
    message = module.add_global(message_str.type_of(), "message")
    message.set_initializer(message_str)
    message_ptr = message.const_bit_cast(context.int8_type().pointer(0))

Note that to build the format string we first have to create a new global
variable, then set its initializer with the constant string and finally cast it
to `int8 *` so that can be passed to `dprintf`, whose prototype is:

.. code-block:: c

    int dprintf(int fd, const char *format, ...);

At this point we have to iterate over all the instructions of the function
containing the generated code (`root`):

.. code-block:: python

    root_function = module.get_named_function("root")
    for basic_block in root_function.iter_basic_blocks():
        for instruction in basic_block.iter_instructions():
            # ...

However, we are not interested in all instructions, but only in calls to
`helper_exception_with_syndrome_*` functions. Therefore, we check the opcode of
the instruction, and, if it's a call, we consider the last operand (which
represents the called function) and check it's name:

.. code-block:: python

    if instruction.instruction_opcode == llvm.Call:

    last_operand_index = instruction.get_num_operands() - 1
        callee = instruction.get_operand(last_operand_index)

        if not callee.name:
            assert(callee.get_num_operands() == 1)
            callee = callee.get_operand(0)

        if callee.name.startswith("helper_exception_with_syndrome_"):
            # ...

Note that the called function is often casted to a slightly different function
type, but we are not interested in this cast. The ``if not callee.name:`` block
handles this situation by moving to the first operand of the cast instruction.

Finally, we've found a location where we want to insert our instrumentation. To
do this, we create a *builder* object, position it right before the call
instruction, emit an instruction to load `r7`, prepare the other arguments and,
finally, emit the call to `dprintf`:

.. code-block:: python

    builder = context.create_builder()
    builder.position_builder_before(instruction)
    load_r7 = builder.build_load(r7, "")
    builder.build_call(dprintf, [two, message_ptr, load_r7], "")

That's all. The last thing left to do is to serialize the new IR to file:

.. code-block:: python

    module.print_module_to_file(sys.argv[2])

Let's now run our script and recompile the code:

.. code-block:: sh

    $ mv hello.ll hello.ll.original
    $ python instrument.py hello.ll.original hello.ll
    $ translate -s hello
    $ ./hello.translated
    45
    45
    983045
    5
    3
    6
    54
    54
    4
    Hello world!
    248

We can compare the result with a QEMU run of the original program:

.. code-block:: sh

    $ qemu-arm -strace hello
    7346 brk(NULL) = 0x00039000
    7346 brk(0x000394b0) = 0x000394b0
    7346 open("/dev/urandom",O_RDONLY) = 3
    7346 read(3,0xf6ffde84,4) = 4
    7346 close(3) = 0
    7346 ioctl(0,21505,-151003688,0,221184,0) = 0
    7346 ioctl(1,21505,-151003688,1,221184,0) = 0
    7346 write(1,0x372a8,13)Hello world!
     = 13
    7346 exit_group(0)

The complete `instrument.py` script is available in `docs/instrument.py`_.

.. _llvmcpy: https://rev.ng/llvmcpy
.. _translate: TranslateUsage.rst
.. _`GeneratedIRReference.rst`: GeneratedIRReference.rst
.. _`docs/instrument.py`: instrument.py
