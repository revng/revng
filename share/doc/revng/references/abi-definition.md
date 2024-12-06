We support adding new ABIs declaratively, available options are explained here.

# Basic explanation

This is a basic explanation as to how `ValueDistributor` works. For extra details, please consult the source code (`lib/ABI/FunctionType/ValueDistributor`).

> **_NOTE:_** The following only applies to ABI with [`ArgumentsArePositionBased`](#argumentsarepositionbased-bool) set to `false`. The selection process for position based ABIs is a lot simpler and is explained in the description of the [option](#argumentsarepositionbased-bool).

First it attempts to allocate a register (or multiple) for a given argument. For packed types[^1] this is only allowed if [`AllowPackedTypesInRegisters`](#allowpackedtypesinregisters-bool) is true.

Also note that sometimes (see [`BigArgumentsUsePointersToCopy`](#bigargumentsusepointerstocopy-bool)) the argument that would not normally be allowed to use registers get transparently replaced by a pointer to a copy of it.

It's important to check whether the type can actually fit, which takes into account [`MaximumGPRsPerAggregateArgument`](#maximumgprsperaggregateargument-unsigned) and friends as well as which arguments from the relevant register list (like [`GeneralPurposeArgumentRegisters`](#generalpurposeargumentregisters-list-of-registers) are still unused.

Next, special consideration are given to "padding" registers in ABIs that require it. This is controlled by [`OnlyStartDoubleArgumentsFromAnEvenRegister`](#onlystartdoubleargumentsfromanevenregister-bool).

Based on all of the previous checks, one of three possible outcomes is selected:
1. The argument is allowed to use and fits into the registers - the "used register" counter advances and the distribution ends.
2. [`ArgumentsCanBeSplitBetweenRegistersAndStack`](#argumentscanbesplitbetweenregistersandstack-bool) is set to true and the argument does not fully fit into the registers - all the remaining registers are marked as used and the left over peace of the argument advances used stack offset. Then some padding is inserted if necessary.
3. The argument uses the stack, with trailing padding inserted if necessary.

The amount of stack padding is determined by [`MinimumStackArgumentSize`](#minimumstackargumentsize-unsigned), but note that it's not always present. For example, [`PackStackArguments`](#packstackarguments-bool) disables such behavior for non-packed[^1] types.

# General options

## `ABI` (`string`)

> **_NOTE:_** this will be renamed into `Name` when we get rid of hard coded ABI names

Specifies which ABI this definition is for.

# Behavior flags

## `ArgumentsArePositionBased` (`bool`)

States whether the ABI is focused on preserving the argument order.

Here's an example function:

```cpp
struct Big; // Is really big, so it can only be passed in memory.
void function(Big, signed, float, unsigned);
```

If the ABI is position based, the arguments would be passed in:
- memory for the `Big` argument (and a pointer to it in the _first_ GPR).
- the _second_ GPR register for the `signed` argument.
- the _third_ vector register for the `float` argument.
- the _forth_ GPR register for the `unsigned` argument.

If the ABI is NOT position based, the arguments would be passed in:
- memory for the `Big` argument.
- the _first_ GPR register for the `signed` argument.
- the _first_ vector register for the `float` argument.
- the _second_ GPR register for the `unsigned` argument.

A typical example of a position-based ABI is `Microsoft_x86_64`, a non-position-based one - `SystemV_x86_64`.

## `OnlyStartDoubleArgumentsFromAnEvenRegister` (`bool`)

States whether an object that needs two GPRs to fit (5-8 bytes on 32-bit architectures and 9-16 bytes on 64-bit systems) is only allowed to start from a register with an even index.

Here's an example function:

```cpp
void function(uint32_t, uint64_t);
```

On a system with 32-bit wide GPRs, the first argument (`uint32_t`) is passed using the first allowed GPR (say `r0`).

The second argument (`uint64_t`) requires two registers to fit, so it's passed using second and third registers if the ABI allows starting double arguments from any register (`r1` and `r2` in this example), or third and forth if it only allows starting them from even registers (`r2` and `r3` in this example, since `r1` is considered an odd register (the count starts from 0, much like C array indexing).

## `ArgumentsCanBeSplitBetweenRegistersAndStack` (`bool`)

States whether the ABI allows a single object that wouldn't fit into a single GPR (9+ bytes on 32-bit systems and 17+ bytes on 64-bit ones) to be partially passed in registers with the remainder placed on the stack if there are not enough registers to fit the entirety of it.

As an example, let's say that there is a big object of type `Big` such that `sizeof(Big)` is equal to 16 bytes. On 32-bit system it would mean having to use four GPRs (`16 == 4 * 4`) to fit it.

Let's look at an ABI that allocates four registers for passing function arguments (`r0-r3`). Then, for a function like

```cpp
void function(uint32_t, Big);
```

the `uint32_t` argument would be passed in the first GPR (`r0`).

But that would also mean that the remaining three available GPRs are not enough to fit the entirety of the `Big` object, meaning it needs to either be split between the registers and the memory, or passed using the stack. That's exactly what this option states.

## `BigArgumentsUsePointersToCopy` (`bool`)

This allows a non-position-based ABI to take advantage of one of the main traits of position-based ABI: ability to use pointers-to-copy to pass big aggregates.

This is one of the defining features of arm64 ABI, which allows it to get the best of both approaches - an ability to pass aggregates by copy in registers from the non-position-based approach while also being able to avoid "wasting" registers on passing big aggregates from the position-based approach.

## `NoRegisterArgumentsCanComeAfterStackOnes` (`bool`)

States whether ABI allows a stack argument (mainly if it's too big to be placed in the registers) to precede other register arguments.

For example, if there is a `Big` struct that has to use the stack, and
a function like

```cpp
void function(Big, uint32_t);
```

if this value is set to true, both argument will be passed on stack, otherwise, only the struct will.

## `AllowPackedTypesInRegisters` (`bool`)

States whether ABI allows packed types[^1] to be passed in registers.

When this option is set to true, such a struct would still be allowed to use register, otherwise it will always use stack.

## `CalleeIsResponsibleForStackCleanup` (`bool`)

Specifies who is responsible for cleaning the stack after the function call. If equal to `true`, it's the callee, otherwise it the caller.

## `FloatsUseGPRs` (`bool`)

Setting this to true disables usage of vector registers entirely, think "soft" float architectures.

## `PackStackArguments` (`bool`)

Most ABIs extend each stack argument to a GPR-sized slot. Setting this option to true, allows them occupy less space by behaving more like fields in a regular struct would.

This option is here exclusively for the apple flavor of the aarch64, since nothing else we support does anything similar.

For example

```cpp
void function(/* enough arguments of all the registers */, uint8_t, uint8_t);
```

When this is set to true, both `uint8_t` will get placed as if they were a struct, that is without any padding in-between. When this is false, both of them will get extended to register size before getting placed, leading to padding.

> **_NOTE:_** This option does not affect types that are packed[^1].

## `TreatAllAggregatesAsPacked` (`bool`)

Treat all the structs as if they were packed[^1].

This means that options that assume regular alignment rules (like `PackStackArguments`) stop applying to them, while others (such as `AllowPackedTypesInRegisters`) start.

Practically this option is mostly used to conditionally ensure [`PackStackArguments`](#packstackarguments-bool) does not apply to structs (see Apple AArch64 ABI for the reference).

# Stack and register limits

## `StackAlignment` (`unsigned`)

States the required alignment of the stack at the point of a function call in bytes.

> **_Note:_** As of now, this value only affects `FinalStackOffset` of the functions in ABIs where callee is responsible for the stack clean up.  That being said, it's still specified for all the ABIs in case we need in the future.

## `MinimumStackArgumentSize` (`unsigned`)

States the minimum possible stack argument size in bytes. When the actual type of an argument is below this value, some padding with be inserted.

## `StackBytesAllocatedForRegisterArguments` (`unsigned`)

States the number of bytes reserved for the callee to be able to "mirror" register arguments on the stack. This value should be set to `0` for any ABI that does not require such space.

In our context, this serves as the starting offset for the first stack argument. As in, it specifies the number of byte on the top of the stack struct that should be ignored.

## `MaximumGPRsPerAggregateArgument` (`unsigned`)

States the maximum number of GPRs available to pass a single aggregate (a struct, a union, etc.) argument, meaning that it can only be passed in the GPRs if `MaximumGPRsPerAggregateArgument` is less than or equal to the number of the registers required to fit the object including padding.

> **_Note:_** If `MaximumGPRsPerAggregateArgument` is equal to 0, it means that the ABI does not allow aggregate arguments to use GPRs.

## `MaximumGPRsPerAggregateReturnValue` (`unsigned`)

States the maximum number of GPRs available to return a single aggregate (a struct, a union, etc.) value, meaning that it can only be returned in the GPRs if `MaximumGPRsPerAggregateReturnValue` is less than or equal to the number of the registers required to fit the object including padding.

> **_Note:_** If `MaximumGPRsPerAggregateReturnValue` is equal to 0, it means that the ABI does not allow aggregate return values to use GPRs.

## `MaximumGPRsPerScalarArgument` (`unsigned`)

States the maximum number of GPRs available to pass a single scalar (`int`, `__int128`, pointer, etc.) argument, meaning that it can only be passed in the GPRs if `MaximumGPRsPerScalarArgument` is less than or equal to the number of the registers required to fit the object.

> **_Note:_** If `MaximumGPRsPerScalarArgument` is equal to 0, it means that the ABI does not allow scalar arguments to use GPRs.

## `MaximumGPRsPerScalarReturnValue` (`unsigned`)

States the maximum number of GPRs available to return a single scalar (`int`, `__int128`, pointer, etc.) value, meaning that it can only be returned in the GPRs if `MaximumGPRsPerScalarReturnValue` is less than or equal to the number of the registers required to fit the object including padding.

> **_Note:_** If `MaximumGPRsPerScalarReturnValue` is equal to 0, it means that the ABI does not allow scalar return values to use GPRs.

# Register lists

## `GeneralPurposeArgumentRegisters` (`list of registers`)

Specifies the list of general purpose registers allowed to be used for passing arguments and the order they are to be used in.

## `GeneralPurposeReturnValueRegisters` (`list of registers`)

Specifies the list of general purpose registers allowed to be used for returning values and the order they are to be used in.

## `VectorArgumentRegisters` (`list of registers`)

Specifies the list of vector registers allowed to be used for passing arguments and the order they are to be used in.

## `VectorReturnValueRegisters` (`list of registers`)

Specifies the list of vector registers allowed to be used for returning values and the order they are to be used in.

## `CalleeSavedRegisters` (`list of registers`)

Specifies the list of registers for which the ABI requires the callee to preserve the value, meaning that when the callee returns, the value of those registers must be the same as it was when the function was called.

# Big return value options

## `ReturnValueLocationRegister` (`register`)

Specifies which register is used for passing the pointer to the memory used for returning big aggregate objects.

For example, for `SystemV_x86_64` ABI it is set to `rdi` (the first argument register), but for `arm64` ABI it is set to a dedicated `x8` register (hence the need for a separate option).

It must be set to `model::Register::Invalid` for ABIs that do not support returning values by "pointer-to-copy" or use stack for it.

## `ReturnValueLocationOnStack` (`bool`)

Specifies whether stack is used to pass the return value location.

This is only relevant if [`ReturnValueLocationRegister`](#returnvaluelocationregister-register) is set to `Invalid`.

## `ReturnValueLocationIsReturned` (`bool`)

Specifies whether functions whose return value is saved in a memory area pointed by an argument passed by the caller (SPTAR) should also return such pointer using the first return value register.

This is only relevant if either [`ReturnValueLocationRegister`](#returnvaluelocationregister-register) or [`ReturnValueLocationOnStack`](#returnvaluelocationonstack-bool) is enabled.

# Type information

## `ScalarTypes` (`list of scalar types`)

This provides a way to introduce some type-specific constraint information to ABI definition, e.g. how types get aligned based on their size.

For each type there are two fields:
- `Size` - the size of the said scalar type
- `AlignedAt` - its alignment. If unset, alignment is assumed to be the same as the size.

## `FloatingPointScalarTypes` (`list of scalar types`)

This provides a way to introduce some type-specific constraint information to ABI definition, e.g. how types get aligned based on their size.

For each type there are two fields:
- `Size` (`unsigned`) - the size of the said scalar type
- `AlignedAt` (`unsigned`) - its alignment. If unset, alignment is assumed to be the same as the size.

-----------

# Glossary

[^1]: by a _packed_ type we explicitly mean such packing that causes a (potentially nested) struct field to start at an offset that is not a multiple of its alignment. You can also think about these as unnaturally aligned types.

For example,

```cpp
struct unnatural __attribute__((packed)) {
  uint8_t first;
  uint16_t second;
};
```

is packed because the second field (the type of which is aligned at two bytes) starts at offset of one byte.
