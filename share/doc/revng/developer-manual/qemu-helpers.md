## Helper Functions

In rev.ng, *helper functions* are C functions originating from QEMU that implement the semantics of complex CPU instructions.

For instance, the x86 `div` instruction has non-trivial semantics (exception on division by zero, quotient overflow, etc.), so QEMU implements it in a C helper called `helper_divb_AL`.

These helpers are compiled to LLVM IR and shipped as part of `libtcg`.

In rev.ng, at *build* time, they undergo significant transformations that prepare them for use during lifting.
This document walks through how helpers evolve, using x86-64 as the driving example.

### The CPU state in QEMU

In QEMU, the entire CPU state lives in a C `struct` called `CPUArchState`.
For x86-64, the relevant fields look like this (from `target/i386/cpu.h`):

```c notest
typedef struct CPUArchState {
    // regs[0] = RAX, regs[1] = RCX....
    target_ulong regs[CPU_NB_REGS];
    target_ulong eip;
    target_ulong eflags;

    /* emulator internal eflags handling */
    target_ulong cc_dst;
    target_ulong cc_src;
    target_ulong cc_src2;
    uint32_t cc_op;
    int32_t df;
    uint32_t hflags; // TB flags, see HF_xxx constants
    uint32_t hflags2;

    /* segments */
    SegmentCache segs[6];
    SegmentCache ldt;
    SegmentCache tr;
    SegmentCache gdt;
    SegmentCache idt;

    target_ulong cr[5]; // cr[0] = CR0
    // ...

    float_status sse_status; // SSE rounding/exception state
    // ...
    ZMMReg xmm_regs[CPU_NB_REGS == 8 ? 8 : 32]; // XMM/YMM/ZMM registers
    ZMMReg xmm_t0; // temporary XMM register
    // ...
} CPUArchState;
```

Every helper takes a `CPUArchState *env` pointer as its first argument and reads or writes the CPU state through it.

### `REVNG_INLINE` and `REVNG_EXCEPTIONAL`

In the QEMU source, helpers are tagged with section attributes that control how rev.ng handles them.
When compiling helpers to LLVM IR, these expand to section attributes:

```c notest
#define REVNG_INLINE __attribute__((section("revng_inline")))
#define REVNG_EXCEPTIONAL __attribute__((section("revng_exceptional")))
```

`REVNG_INLINE` marks helpers whose body rev.ng will inline at a certain point in the [pipeline](../references/pipeline/).
Helpers *not* tagged with `REVNG_INLINE` are kept as opaque calls.

`REVNG_EXCEPTIONAL` marks helpers that are considered to be "exceptional cases", like division by 0 or an invalid memory access. At a certain point in the pipeline, we assume these situations don't happen. This enables us to remove these calls (and all the code they postdominate) and emit better looking code.

### Example helpers

Let's now see some example helper functions, as they are in QEMU.

Let's consider `helper_clts`, a `REVNG_INLINE` helper implementing the x86 `clts` instruction (clear Task-Switch flag in CR0).
The TS (Task-Switch) flag is bit 3 of CR0.
The CPU sets it on every hardware task switch; when set, any FP or SSE instruction traps with a `#NM` (Device Not Available) exception, allowing the OS to lazily save and restore floating-point state.
The `clts` instruction clears this flag, and the helper mirrors it into the internal `hflags` register:

```c notest
void helper_clts(CPUX86State *env) REVNG_INLINE
{
    env->cr[0] &= ~CR0_TS_MASK;
    env->hflags &= ~HF_TS_MASK;
}
```

Let's then consider `helper_divb_AL`, a `REVNG_INLINE` helper implementing the x86 `div r/m8` instruction (unsigned byte division).
This instruction divides `AX` (the low 16 bits of `RAX`, the *implicit* dividend) by an 8-bit operand (the *explicit* divisor, passed as `t0`).
The quotient is stored in `AL` and the remainder in `AH` (both packed back into `RAX`).
A `#DE` (divide error) exception is raised if the divisor is zero or if the quotient exceeds 0xFF:

```c notest
void helper_divb_AL(CPUX86State *env, target_ulong t0) REVNG_INLINE
{
    unsigned int num, den, q, r;

    num = (env->regs[R_EAX] & 0xffff);
    den = (t0 & 0xff);
    if (den == 0) {
        raise_exception_ra(env, EXCP00_DIVZ, GETPC());
    }
    q = (num / den);
    if (q > 0xff) {
        raise_exception_ra(env, EXCP00_DIVZ, GETPC());
    }
    q &= 0xff;
    r = (num % den) & 0xff;
    env->regs[R_EAX] = (env->regs[R_EAX] & ~0xffff) | (r << 8) | q;
}
```

On the exceptional paths, `raise_exception_ra` is a `REVNG_EXCEPTIONAL` function.

Finally, `helper_write_eflags` is a helper *without* `REVNG_INLINE`.

```c notest
void helper_write_eflags(CPUX86State *env, target_ulong t0,
                         uint32_t update_mask)
{
    cpu_load_eflags(env, t0, update_mask);
}
```

### The original helpers in LLVM IR

The original helpers live in `share/libtcg/libtcg-helpers-x86_64.bc`.

!!! tip

    In the following snippets we apply the LLVM `-sroa` (Scalar Replacement of Aggregates), `-instcombine` (instruction combining) and `-dce` (dead code elimination) passes to the IR.
    This eliminates stack allocations, promotes local variables to SSA values, folds redundant GEPs and removes dead instructions, making the IR much easier to read.

We also define a `pretty` shell function that strips LLVM attribute-group references, metadata annotations and trailing comments from the textual IR:

```bash
$ ROOT="$(dirname "$(dirname "$(which revng)")")"
$ pretty() { sed "s/ #[0-9]*//; s/ ![^ ]*//g; s/;.*//"; }
$ revng opt -strip-debug -sroa -instcombine -dce -S \
    "$ROOT/share/libtcg/libtcg-helpers-x86_64.bc" \
    -o libtcg-helpers-x86_64-optimized.S
$ cat libtcg-helpers-x86_64-optimized.S \
    | sed -n "/^define void @helper_clts/,/^}/p" \
    | pretty
define void @helper_clts(ptr noundef %0) section "revng_inline" {
  %2 = getelementptr inbounds %struct.CPUArchState, ptr %0, i64 0, i32 15
  %3 = load i64, ptr %2, align 8
  %4 = and i64 %3, u0xfffffff7
  store i64 %4, ptr %2, align 8
  %5 = getelementptr inbounds %struct.CPUArchState, ptr %0, i64 0, i32 8
  %6 = load i32, ptr %5, align 16
  %7 = and i32 %6, u0xfffff7ff
  store i32 %7, ptr %5, align 16
  ret void
}
```

The function takes a `%struct.CPUArchState` pointer `%0` (the `env` argument) and navigates into it using `getelementptr`:

- Field index 15 is `cr[0]` (the first element of the `cr` array). It loads the value, clears the TS bit (`and` with `0xfffffff7`), and stores it back.
- Field index 8 is `hflags`, an `i32`. It clears the `HF_TS_MASK` bit.

Now let's look at `helper_divb_AL`:

```bash
$ cat libtcg-helpers-x86_64-optimized.S \
    | sed -n "/^define void @helper_divb_AL/,/^}/p" \
    | pretty \
    | head -16
define void @helper_divb_AL(ptr noundef %0, i64 noundef %1) section "revng_inline" {
  %3 = load i64, ptr %0, align 16
  %4 = trunc i64 %3 to i32
  %5 = and i32 %4, u0xffff
  %6 = trunc i64 %1 to i32
  %7 = and i32 %6, 255
  %8 = icmp eq i32 %7, 0
  br i1 %8, label %9, label %10

9:
  call void @raise_exception_ra(ptr noundef nonnull %0, i32 noundef 0, i64 noundef 0)
  unreachable

10:
  %11 = udiv i32 %5, %7
  %12 = icmp ugt i32 %11, 255
```

Since `regs` is the first field of `CPUArchState` and `R_EAX` is index 0, `instcombine` folds the GEP chain away and loads directly from `%0`.
The function loads `RAX`, masks the low 16 bits (`num = env->regs[R_EAX] & 0xffff`), truncates the divisor `%1` to 8 bits, and checks for division by zero.
Notice the call to `raise_exception_ra` followed by `unreachable`.

All these accesses go through the `env` struct pointer.

This is a problem for rev.ng: during lifting, the CPU state is not a struct in memory but a set of independent global variables called *CSVs* (CPU State Variables).
The build-time passes solve exactly this.

### Build-time processing

At build time, the helpers undergo a chain of transformations.
Each step produces a new bitcode file derived from the previous one:

1. QEMU's C helper sources are compiled to LLVM IR, producing the *original* helpers in `share/libtcg/libtcg-helpers-x86_64.bc`. These still access the CPU state through the `env` struct pointer.

2. The *full* module (`share/revng/libtcg-helpers-full-x86_64.bc`) is derived from the original by running the `fix-helpers` pass, which replaces every `env` struct access with an access to the corresponding CSV. All helper bodies are present. For x86-64 this is ~53 MB.

3. The *to-inline* module (`share/revng/libtcg-helpers-to-inline-x86_64.bc`) is derived from the full module by stripping the bodies of all helpers *not* tagged with `REVNG_INLINE`. Only `REVNG_INLINE` helpers retain their bodies. ~3 MB.

4. The *declarations-only* module (`share/revng/libtcg-helpers-declarations-only-x86_64.bc`) is also derived from the full module, but goes further: *all* helper bodies are stripped, leaving only declarations with CSV access metadata.

The size difference matters: linking unnecessary code slows down the pipeline without bringing any benefit.
Each pipeline stage links only what it needs:

* the `lift` step links the *declarations-only* module — it only needs function signatures and CSV access metadata to emit calls;
* `inline-helpers` links the *to-inline* module and inlines the `REVNG_INLINE` bodies into the lifted code;
* `recompile` needs every helper implementation, so it links the large *full* module.

### The full helpers

Let's look at `helper_clts` after the `fix-helpers` transformation:

```bash
$ revng opt -strip-debug -instcombine -dce -S \
    "$ROOT/share/revng/libtcg-helpers-full-x86_64.bc" \
    -o libtcg-helpers-full-x86_64-optimized.S
$ cat libtcg-helpers-full-x86_64-optimized.S \
    | sed -n "/^define void @helper_clts/,/^}/p" \
    | pretty
define void @helper_clts(ptr noundef %0) section "revng_inline" {
  %2 = load i64, ptr @_state_0x2968, align 8
  %3 = and i64 %2, u0xfffffff7
  store i64 %3, ptr @_state_0x2968, align 8
  %4 = load i32, ptr @_state_0x2870, align 4
  %5 = and i32 %4, u0xfffff7ff
  store i32 %5, ptr @_state_0x2870, align 4
  ret void
}
```

Instead of loading and storing through the `env` struct pointer, the helper now reads from and writes to *CSVs*: `@_state_0x2968` for `cr[0]` and `@_state_0x2870` for `hflags`.
The dead `getelementptr` and `ptrtoint` instructions left over from the annotation have been eliminated by `instcombine`.

The hex number in a CSV name is the byte offset of the field within the broader `CPUState` struct (which wraps `CPUArchState`).
Well-known registers get human-readable names instead.
For instance, `helper_divb_AL` after annotation:

```bash
$ cat libtcg-helpers-full-x86_64-optimized.S \
    | sed -n "/^define void @helper_divb_AL/,/^}/p" \
    | pretty \
    | head -10
define void @helper_divb_AL(ptr noundef %0, i64 noundef %1) section "revng_inline" {
  %3 = load i64, ptr @_rax, align 8
  %4 = trunc i64 %3 to i32
  %5 = and i32 %4, u0xffff
  %6 = trunc i64 %1 to i32
  %7 = and i32 %6, 255
  %8 = icmp eq i32 %7, 0
  br i1 %8, label %9, label %12

9:
```

Here `@_rax` (the CSV for the `rax` register) replaces the `getelementptr` + `load` through `env->regs[R_EAX]`.

#### Multiple CSVs for a single access

In `helper_clts` and `helper_divb_AL`, each memory access through `env` targets exactly one field. The `fix-helpers` pass can replace each access with a direct load/store of the corresponding CSV.

Not all helpers are that simple. Consider `helper_addsd`, which implements the x86 `addsd` instruction (add scalar double-precision floating-point).
It takes three `ZMMReg *` pointer arguments (`d`, `v`, `s`) that can each point to *any* XMM register in `xmm_regs[0..31]` or the temporary `xmm_t0`:

```c notest
void helper_addsd(CPUX86State *env, ZMMReg *d, ZMMReg *v, ZMMReg *s)
        REVNG_INLINE
{
    d->ZMM_D(0) = float64_add(v->ZMM_D(0), s->ZMM_D(0),
                               &env->sse_status);
    d->ZMM_Q(1) = v->ZMM_Q(1);
}
```

In the original IR, the accesses are simple pointer dereferences (`load i64, ptr %2`).
But in the full module, `fix-helpers` cannot replace them with a single CSV — the same pointer could refer to any of 33 different registers.
Instead, it emits a `switch` on the pointer value (which is the byte offset of the register within `CPUState`) to dispatch to the correct CSV:

```bash
$ cat libtcg-helpers-full-x86_64-optimized.S \
    | sed -n "/^define void @helper_addsd/,/^}/p" \
    | pretty \
    | sed -n '1,12p'
define void @helper_addsd(ptr noundef %0, ptr noundef %1, ptr noundef %2, ptr noundef %3) section "revng_inline" {
  %5 = ptrtoint ptr %2 to i64
  switch i64 %5, label %39 [
    i64 u0x2f10, label %6
    i64 u0x2f50, label %7
    i64 u0x2e90, label %8
    i64 u0x2e50, label %9
    i64 u0x3010, label %10
    i64 u0x2b10, label %11
    i64 u0x3150, label %12
    i64 u0x2bd0, label %13
    i64 u0x3050, label %14
```

Each case label corresponds to the offset of a different XMM register (e.g. `0x2b10` is `xmm_regs[0]`, `0x2b50` is `xmm_regs[1]`, etc., stepping by 64 bytes).
In each case, the load is replaced by a direct access to the corresponding CSV.

### The to-inline helpers

The *to-inline* variant keeps function *bodies* only for helpers marked with `REVNG_INLINE`.
All other helpers are dropped entirely.

For example, `helper_clts` (which is `REVNG_INLINE`) still has its full definition:

```bash
$ revng opt -strip-debug -S \
    "$ROOT/share/revng/libtcg-helpers-to-inline-x86_64.bc" \
    -o libtcg-helpers-to-inline-x86_64-optimized.S
$ cat libtcg-helpers-to-inline-x86_64-optimized.S \
    | grep "^define.*@helper_clts" \
    | pretty
define void @helper_clts(ptr noundef %0) section "revng_inline" {
```

We can see the counts: the *to-inline* module has a small number of definitions (the `REVNG_INLINE` helpers):

```bash
$ echo "Definitions:"
$ cat libtcg-helpers-to-inline-x86_64-optimized.S \
    | grep -c "^define"
Definitions:
363
```

### The declarations-only helpers

The *declarations-only* variant (`share/revng/libtcg-helpers-declarations-only-x86_64.bc`) goes one step further: *no* helper has a body.
Every function, including `REVNG_INLINE` ones, is a bare declaration.

```bash
$ revng opt -strip-debug -S \
    "$ROOT/share/revng/libtcg-helpers-declarations-only-x86_64.bc" \
    -o libtcg-helpers-declarations-only-x86_64-optimized.S
$ cat libtcg-helpers-declarations-only-x86_64-optimized.S \
    | grep "^declare.*@helper_clts" \
    | pretty
declare void @helper_clts(ptr noundef) section "revng_inline"
```

The module has 0 definitions and 1089 declarations:

```bash
$ echo "Definitions:"
$ cat libtcg-helpers-declarations-only-x86_64-optimized.S \
    | { grep -c "^define" || true; }
Definitions:
0
$ echo "Declarations:"
$ cat libtcg-helpers-declarations-only-x86_64-optimized.S \
    | grep "^declare" | grep -c "helper_"
Declarations:
1089
```

### CSV access metadata

All helper variants carry `!revng.csvaccess.offsets.load` and `!revng.csvaccess.offsets.store` metadata on every helper.
This metadata records which CSVs a helper reads and which it writes, *even when no body is available*.

For instance, in the *declarations-only* module, `helper_write_eflags` has no body, yet its declaration carries the metadata:

```bash
$ cat libtcg-helpers-declarations-only-x86_64-optimized.S \
    | grep "^declare.*@helper_write_eflags"
declare !revng.csua !299 !revng.csvaccess.offsets.load !303 !revng.csvaccess.offsets.store !305 !revng.tags !13 void @helper_write_eflags(ptr noundef, i64 noundef, i32 noundef) #0
```

The `!303` and `!305` are references to metadata nodes defined at the end of the module.
Resolving them reveals the actual CSV lists:

```bash
$ cat libtcg-helpers-declarations-only-x86_64-optimized.S \
    | grep -E "^!(303|304|305|306) ="
!303 = !{i32 0, !304}
!304 = !{!"_state_0x2848"}
!305 = !{i32 0, !306}
!306 = !{!"_cc_src", !"_state_0x286c", !"_state_0x2848", !"_cc_op"}
```

Each CSV access metadata node is a tuple `!{i32 0, !<csv-list>}` where the second element lists the CSV names.
Here, `helper_write_eflags` reads `_state_0x2848` (the `eflags` field) and writes four CSVs: `_cc_src`, `_state_0x286c` (`df`), `_state_0x2848` (`eflags`), and `_cc_op`.

This is critical for the *declarations-only* module: analyses can determine the side effects of a helper call purely from metadata, without inspecting a body that is not present.

### Usage in the pipeline

Each variant is used by a different stage of the rev.ng [pipeline](../references/pipeline/).

#### Helpers in the `lift` artifact

At lift time, the *declarations-only* module is linked in.
The lifter only needs function signatures to emit calls; it does not need bodies.
The CSV access metadata is enough to inform analyses about what each helper reads and writes.

In the following, we create a minimal binary that divides the first argument (`rdi`) by the low byte of the second (`sil`), which triggers a call to `helper_divb_AL`.
The model tells rev.ng the binary's architecture, memory layout, and function prototypes.
We declare a single function at `0x400000` with a two-argument prototype:

```yaml title="model.yml"
---
# C prototype: uint64_t func(uint64_t arg0, uint64_t arg1)
Architecture: x86_64
DefaultABI: SystemV_x86_64
Segments:
  - StartAddress: "0x400000:Generic64"
    VirtualSize: 6
    StartOffset: 0
    FileSize: 6
    IsReadable: true
    IsWriteable: false
    IsExecutable: true
Functions:
  - Entry: "0x400000:Code_x86_64"
    Prototype:
      Kind: DefinedType
      Definition: "/TypeDefinitions/0-CABIFunctionDefinition"
TypeDefinitions:
  - Kind: CABIFunctionDefinition
    ABI: SystemV_x86_64
    ID: 0
    Arguments:
      - Index: 0
        Type:
          Kind: PrimitiveType
          PrimitiveKind: Unsigned
          Size: 8
      - Index: 1
        Type:
          Kind: PrimitiveType
          PrimitiveKind: Unsigned
          Size: 8
    ReturnType:
      Kind: PrimitiveType
      PrimitiveKind: Unsigned
      Size: 8
...
```

```bash
$ printf '\x89\xf8\x40\xf6\xf6\xc3\x90' > div-binary
$ objdump -D -Mintel,x86-64 -b binary -m i386:x86-64 div-binary

div-binary:     file format binary


Disassembly of section .data:

0000000000000000 <.data>:
   0: 89 f8                 mov    eax,edi
   2: 40 f6 f6              div    sil
   5: c3                    ret
   6: 90                    nop
```

```bash silent
$ revng artifact lift div-binary --model model.yml -o module.bc
```

Let's inspect the basic block.
The `mov eax, edi` copies the first argument into the accumulator; then `div sil` divides it by the low byte of the second argument:

```bash
$ revng opt -strip-debug -S module.bc \
    | sed -n '/^"bb.0x400000:Code_x86_64":/,/^$/p' \
    | pretty
"bb.0x400000:Code_x86_64":
  call void (ptr, i64, i32, i32, ptr, ...) @newpc(ptr nonnull @"revng.const.0x400000:Code_x86_64", i64 2, i32 1, i32 0, ptr null)
  %5 = load i64, ptr @_rdi, align 8
  %6 = and i64 %5, u0xffffffff
  store i64 %6, ptr @_rax, align 8
  call void (ptr, i64, i32, i32, ptr, ...) @newpc(ptr nonnull @"revng.const.0x400002:Code_x86_64", i64 3, i32 0, i32 0, ptr null)
  %7 = load i64, ptr @_rsi, align 8
  call void @helper_divb_AL(ptr nonnull inttoptr (i64 u0x27c0 to ptr), i64 %7)
  store i1 false, ptr @cpu_loop_exiting, align 1
  call void (ptr, i64, i32, i32, ptr, ...) @newpc(ptr nonnull @"revng.const.0x400005:Code_x86_64", i64 1, i32 0, i32 0, ptr null)
  %8 = load i64, ptr @_rsp, align 8
  %9 = inttoptr i64 %8 to ptr
  %10 = load i64, ptr %9, align 1
  %11 = add i64 %8, 8
  store i64 %11, ptr @_rsp, align 8
  store i64 %10, ptr @_rip, align 8
  br label %anypc,
```

The `mov eax, edi` becomes `load @_rdi` → `and` (zero-extend to 32-bit) → `store @_rax`.
The `div sil` becomes `call void @helper_divb_AL(env, i64 %7)` where the env pointer is folded to a constant `inttoptr` and `%7` is the value of `@_rsi`.
The `ret` pops the return address from `@_rsp` into `@_rip`.

#### Helpers in the `enforce-abi` artifact

At the `enforce-abi` stage, the *to-inline* module is linked.
The `inline-helpers` pass walks each isolated function, finds calls to functions in `section "revng_inline"`, and inlines them in a fixed-point loop.

```bash silent
$ revng artifact enforce-abi div-binary --model model.yml -o enforced.bc
```

Let's look at the isolated function.

```bash
$ revng opt -strip-debug -S enforced.bc \
    | sed -n "/^define.*@local_0x400000_Code_x86_64/,/^}/p" \
    | pretty \
    | sed -n "1p; /and i64.*u0xffff$/,/helper_divb_AL.exit:/p"
define i64 @local_0x400000_Code_x86_64(i64 %rdi_x86_64, i64 %rsi_x86_64) {
  %27 = and i64 %26, u0xffff
  %28 = trunc i64 %27 to i32
  %29 = and i64 %25, 255
  %30 = trunc i64 %29 to i32
  %31 = icmp eq i32 %30, 0
  br i1 %31, label %32, label %33

32:
  unreachable

33:
  %34 = udiv i32 %28, %30
  %35 = icmp ugt i32 %34, 255
  br i1 %35, label %36, label %37

36:
  unreachable

37:
  %38 = and i32 %34, 255
  %39 = urem i32 %28, %30
  %40 = and i32 %39, 255
  %41 = load i64, ptr %_rax, align 8
  %42 = and i64 %41, u0xffffffffffff0000
  %43 = shl i32 %40, 8
  %44 = zext i32 %43 to i64
  %45 = or i64 %42, %44
  %46 = zext i32 %38 to i64
  %47 = or i64 %45, %46
  store i64 %47, ptr %_rax, align 8
  br label %helper_divb_AL.exit

helper_divb_AL.exit:
```

The `call void @helper_divb_AL(...)` is gone, its body has been inlined.
Since `remove-exceptional-functions` is part of the `enforce-abi` pipeline, the `raise_exception_ra` calls (which are `REVNG_EXCEPTIONAL`) have already been replaced with `unreachable`.

Compare this with the C source: the `udiv`/`urem` implement the division, `%_rax` is the accumulator, and the two `unreachable` blocks (labels 12 and 16) are where `raise_exception_ra` used to be (division-by-zero and quotient-overflow checks).

Running `-simplifycfg` eliminates the `unreachable` blocks, turning the error conditions into `llvm.assume` intrinsics.
These `llvm.assume` calls are later removed by the `remove-llvmassume-calls` pass (which runs as part of the `segregate-stack-accesses` step).
Adding `-dce` cleans up the remaining dead instructions:

```bash
$ revng opt -strip-debug -simplifycfg -remove-llvmassume-calls -dce -S enforced.bc \
    | sed -n "/^define.*@local_0x400000_Code_x86_64/,/^}/p" \
    | pretty \
    | sed -n "1p; /and i64.*u0xffff$/,/store i64.*%_rax/p"
define i64 @local_0x400000_Code_x86_64(i64 %rdi_x86_64, i64 %rsi_x86_64) {
  %27 = and i64 %26, u0xffff
  %28 = trunc i64 %27 to i32
  %29 = and i64 %25, 255
  %30 = trunc i64 %29 to i32
  %31 = udiv i32 %28, %30
  %32 = and i32 %31, 255
  %33 = urem i32 %28, %30
  %34 = and i32 %33, 255
  %35 = load i64, ptr %_rax, align 8
  %36 = and i64 %35, u0xffffffffffff0000
  %37 = shl i32 %34, 8
  %38 = zext i32 %37 to i64
  %39 = or i64 %36, %38
  %40 = zext i32 %32 to i64
  %41 = or i64 %39, %40
  store i64 %41, ptr %_rax, align 8
```

The exceptional calls and dead code are completely gone.
What remains is a clean straight-line byte division: load RAX, divide, store quotient and remainder back into RAX.

#### Helpers in the `recompile` artifact

At recompile time, the *full* module is linked in.
Since the recompiler produces native code for *all* helper calls (including non-inline ones), it needs every helper definition — hence the large (~53 MB) *full* module.
