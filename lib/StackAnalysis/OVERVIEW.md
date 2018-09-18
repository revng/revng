This document describes from an high level point of view the stack analysis and
its components.

# The `StackAnalysis` pass

The `StackAnalysis` pass is where everything begins. Its `run` method does the
following:

* It keeps an instance of the `Cache` class, which holds the results of the
  analyses for each function analyzed so far. The `Cache` also identifies, for
  each function call, where the return address is stored (i.e., the link
  register or the top of the stack) and what's the most common place to store
  the return address (an information that will be employed for functions that
  have no direct calls).
* It identifies the function entry points. They are divided in two sets, one for
  the entry points that are highly likely to represent a function (in
  particular, those that are target of a direct call) and then the rest of
  possible candidates.
* It runs the analysis on the first set of functions.
* It runs the analysis on the functions of the second set whose entry basic
  block has not been identified as being part of a function of the first set.
* It collects the results of the analysis in `ResultsPool`.
* It produces the final version of the results contained in `ResultsPool`,
  obtaining a `FunctionsSummary` object.

# The `InterproceduralAnalysis`

What we called *the analysis* is actually `InterproceduralAnalysis`. A run of
the `InterproceduralAnalysis` analyzes a single function and collects the
results of the analysis in a `ResultsPool` object. The `ResultsPool` contains
intermediate information that will be then merged into the final object exposed
by the analysis, i.e., `FunctionsSummary`. Note that the `Cache` is used
exclusively for handling function calls during the intraprocedural analysis, it
has nothing to do with the collection of the final results, which is handled by
`ResultsPool` and `FunctionsSummary`.

To perform the analysis, `InterproceduralAnalysis` performs one or more
`Intraprocedural::Analysis`, starting from the entry point, but, if necessary,
analyzing all the functions in the call graph starting from the original entry
point.

In practice, each time an `Intraprocedural::Analysis` meets a call site
targeting a function not already analyzed (i.e., not present in `Cache`), the
`Intraprocedural::Analysis` is suspended and the control is returned to the
`InterproceduralAnalysis` that will start a new `Intraprocedural::Analysis` for
the callee. The list of suspended `Intraprocedural::Analysis` is recorded on a
stack, whose top element is the currently running analysis.

The `InterproceduralAnalysis` also handles recursion. In case the
`Intraprocedural::Analysis` has been suspended due to a uncached function call,
the interprocedural part will detect if the callee is a function already present
in the stack of the currently in-progress analyses and, in such case, it will
inject a temporary top entry in the cache. The intraprocedural analysis can now
proceed with a safe (and pessimistic) assumption.

Once the analysis of the recursive function is terminated, as usual, it is
recorded in the cache. If the result is different from the cached one, the
analysis is restarted from scratch, but in this case the recursive call site
will use the more accurate information provided by the last analysis. This
process is repeated until a fixed point is reached, i.e., the result of the
analysis matches the one already present in the cache.

# The intraprocedural analysis (stack analysis)

The core goal of `Intraprocedural::Analysis` is to analyze how the registers are
used in a function and the extension of the function itself.

In particular, the intraprocedural analysis can:

* detect if a register is a callee-saved register in a function;
* if an indirect jump is a return instruction;
* if a function misbehaves with the stack and should therefore be considered a
  "fake function", i.e., a function to inline in the callers;

The intraprocedural analysis (in the `StackAnalysis::Intraprocedural`
namespace) is an instance of the `MonotoneFramework` class, and, as such, it has
a couple of interesting parts: the *lattice* and the *transfer function*.

The intraprocedural analysis is a forward analysis.

## The lattice

The lattice is defined by the `Element` class. The `Element` class is basically
a container for a set of `AddressSpace`. An `Element` can have 0 or 3 address
spaces.

The first `AddressSpace` (or "alias domain") represents the CPU state (`CPU`),
the second the global variables and the heap (`GLB`), while the third is for the
the stack frame of the current function (`SP0`).

An `AddressSpace` tracks its content in the form of "slots" (`ASSlot`), i.e., a
pair of the identifier of an address space and an offset within it. For the
`CPU` address space, the offset represent the index of the CSV.

The `GLB` address spaces is also used to track constants. This means that the
constant 42 will be represented as an `ASSlot` relative to `GLB` with offset 42
(`GLB+42`).

Note that our analyses currently ignore overlapping slots. Note also that
overlapping slots are not possible in the `CPU` address space (CSVs never
alias).

An `AddressSpace` associates to each slot a `Value`. A `Value` is composed by
two `ASSlot` fields: a "direct content", i.e., the content of the slot in a
certain program point (according to our analysis) and a "tag". A tag represents
the fact that we are not able to track the actual content of that slot
statically, but we know that it contains the value that another slot contained
at the entry of the function. The tag is useful to represent, e.g., the
information that a certain stack slot (where the value of a callee-saved
register is saved) contains the initial value of a register, or that an indirect
jump is targeting a `Value` representing the initial value, e.g., of the link
register. In the latter case, we basically proved that the indirect jump is
actually a return instruction.

The intraprocedural analysis associates to each SSA value an element of the
lattice. However, only the lattice element at the end of the basic block
currently being processed is of our interest (so that it can be propagated to
its successors).

## The transfer function

The transfer function of the intraprocedural analysis handles mainly the
following types of instructions:

* `StoreInst`: a store has an address and a value to store, they will be both
  associated to a `Value`. If the address `Value` has a direct content, the
  `AddressSpace` associated to the direct content `ASSlot` will be updated at
  the appropriate offset.
* `LoadInst`: if the `Value` associated to the pointer operand has a direct
  content, we will look up in the corresponding address space at the
  corresponding offset if have recorded a `Value`. If so, the result is that
  `Value`. Otherwise this means that we are not aware of any store instruction
  targeting that address, therefore, the resulting `Value` won't have a direct
  content but just a tag representing the loaded address.
* `TerminatorInst`: terminator instructions go through a classification
  depending on the context in which they are performed. For instance, an
  indirect jump might be detected as a return instruction if it's jumping to a
  `Value` tagged with the link register `ASSlot` *and* the stack is no higher
  than how it was at entry of the function.

  A `TerminatorInst` can also represent a function call, in such case, its
  callee (if not indirect) is looked up into the `Cache`, if it's available the
  result of its analysis replace the current state of the `AddressSpace`s,
  otherwise the intraprocedural analysis is suspended and the control is
  returned to the interprocedural part as described above.

Other basic instructions are handled in the straightforward way, e.g., addition.

# The ABI analysis

As part of the finalization of the results of the intraprocedural analysis
(`Analysis::createSummary`), the ABI analysis is performed. The goal of the ABI
analysis is to detect arguments and return values of function and function
calls.

## The ABI IR

The ABI analysis is performed on a custom IR, the ABI IR, which is produced by
the intraprocedural analysis during its execution. The main motivating reason
for having this IR is to facilitate debugging and, most importantly, being able
to perform backward analyses easily.

The ABI IR is quite simple: the `ABIIRFunction` is a container of
`ABIIRBasicBlock`s which in turn are containers for `ABIIRInstruction`s.  Each
basic block has links to its successors and predecessors.

An `ABIIRInstruction` can be of the following types:

* `Load`: a read from an `ASSlot`;
* `Store`: a write to an `ASSlot` (what is being written, is of no interest);
* `DirectCall`: a function call for which a result of a previously run ABI
  analysis is available.
* `IndirectCall`: a function call about which nothing is known.

## The analyses

The `FunctionABI` class is responsible for performing all the analyses
concerning the ABI. In particular, the `analyze` method performs two sets of
analyses: the first set are forward, while the second one is backward.

Each set can be further divided into two groups: the analyses concerning the
*function* itself and analyses concerning the *function calls*.

Each function-level analysis starts with a `Default` instance, then, each time a
memory access to a certain CSV is met, the `Default` instance is cloned and
associated to that CSV, whose analysis then proceeds independently.

The same holds for function call-level analyses, with the distinction that the
same "lazy" instantiation of a set of analyses happens also each time a call
site is met.

The set of function-level analyses contain all the function call-level analyses
too so that their result can be used by the calling function performing function
call-level analyses (as if they were inlined).

Each analysis is wrapped in an `Inhibitor` class, which, as the name suggests,
is used to inhibit the wrapped analysis from applying the transfer function
while walking the IR. Function-level analyses are never inhibited. Function
call-level analyses, instead, start as inhibited and, once the corresponding
function call is met, they are enabled. This helps us to simulate the beginning
of the analysis in that point. The transfer function of a function call analysis
reaching for the second time the function call is the unknown function call
transfer function.

For the list of forward, backward, function and function call analyses, consult
the source code and the `.dot` files used to generate them.

# Merging the results

At the end of all of the analyses, the `ResultsPool` object will contain a
summary of all the recovered information such as the basic block composing a
function, the type of function, the status of each registers in terms of being
an argument or a return value for each function and for each function call and
so on.

These information, and, specifically the last two pieces of information, have to
be merged together in order to produce more accurate information or identify
contradictions. This step is performed in `ResultsPool::finalize`.

Basically the idea is that if for a certain register we have a `Yes` from a call
site and a `NoOrDead` from the function itself, we will produce as a final
information for the call site `Dead`. We will produce `Dead` for the function
too only in case *all* of the call sites agree.

# Ad-hoc handling of peculiar situations

For various reasons, in part concerning our code generation pipeline and in part
due to certain practices in compiler backends, we have to handle certain
situations in an ad-hoc way to avoid mistakes.

In the following we will discuss the situations we currently handle.

## Fake functions

Consider the following example of ARM code:

    _start:
        push {lr}
        bl prologue
        ldr r0, [r0]
        b epilogue

    prologue:
        push {r0}
        push {r1}
        bx lr

    epilogue:
        pop {r1}
        pop {r0}
        pop {lr}
        bx lr

The compiler (or the developer) decided to outline the function prologue and
epilogue, likely for code size reduction reasons. In this situation we don't
really want to consider `prologue` and `epilogue` as standalone functions, for
two reasons: they manipulate the stack in weird ways and prevent us from
identifying callee-saved registers.

The jump to the epilogue is not a problem since it will automatically considered
part of each function jumping there. On the other hand, we need to make sure
that we can correctly identify `prologue` as a *fake* function, and, therefore,
inline it in the caller.

To do this, we can note that at the end of the "function" the stack is higher
than it was at the beginning. No sane function call would allow this. Therefore,
we mark `prologue` as a fake function.

Note, on the other hand, than having a stack *lower* than it was at the
beginning is allowed, since certain caller conventions mandate to the callee the
cleanup of stack arguments (e.g., the Windows PASCAL calling convention).

As a consequence, when we analyze a terminator during the intraprocedural
analysis (`handleTerminator` method), when we meet an instruction that jumps to
the initial content of the link register, we understand it's a return, but if we
see that the stack is higher than it initially was, we mark the functions as
fake and resume the analysis of the caller (if any), in which we will inline the
function call.

Consider now the following a variation of the previous snippet:

    _start:
        push {lr}
        add sp,sp,-8
        bl prologue
        ldr r0, [r0]
        b epilogue

    prologue:
        str r1,[sp,0]
        str r0,[sp,4]
        bx lr

    epilogue:
        pop {r1}
        pop {r0}
        pop {lr}
        bx lr

In this case, the stack pointer is not touched by the `prologue`
function. Therefore, the previous criteria is not effective. In this case we
observe another fact: the `prologue` function writes in `SP0+0` and `SP0+4`,
which seem to be stack argument. This is fine, however we keep track of
this. The next thing we observe is that the same stack slots are *read* by the
caller (in the `epilogue` basic block). Under our assumptions, this is not
allowed, since no return value is passed (directly) on the stack and stack
arguments are no longer valid after the function returns.  Therefore, we mark
the called function as fake.

This analysis is performed by the `StackAnalysis::IncoherentCallsAnalysis`
analysis, which is performed on the ABI IR. The analysis is triggered by the
`findIncoherentFunctions` function in the `createSummary` method of the
intraprocedural analysis, after the ABI analysis has been run.

## Forwarded arguments

Consider the following snippet of x86-64 assembly:

    push_pop:
        push rax
        pop rdx
        ret

This code is sometimes emitted by the compiler with the only goal of growing and
decreasing the stack height. The `rax` and `rdx` registers do not contain
anything meaningful.

The problem with this snippet is that, according to our analyses, `rax` is an
argument and `rdx` is a return value, while, obviously this is not the case.

To detect this situation, the `IntraproceduralFunctionSummary`, which holds the
final result of the intraprocedural analysis of a functions, has a `process`
method the pattern matches it: if a return value is tagged with the initial
value of different register and that value is also stored in a stack slot, we
mark it as a forwarded argument.

Statements about such registers in terms of being arguments/return values will
be weakened.

## Identity loads

Identity loads are a particular type of load instructions. Consider the
following x86-64 pseudo code:

    a = rax & 0xffff0000
    b = 0xaaaa
    rax = a | b

This snippet is the result of writing only the lowest 16 bits of `rax` (with
`0xAAAA`), however, from our point of view we have a read of `rax` before any
write and would, therefore, consider it an argument. This load is an identity
load, a load whose value will end up as is in itself.

Identity loads are ignored completely. They are not even part of the ABI IR.

The `Cache` is in charge to identify and keep track of identity loads
(`Cache::identifyIdentityLoads`).
