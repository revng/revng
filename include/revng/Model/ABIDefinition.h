#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <string>
#include <vector>

#include "revng/Model/Architecture.h"
#include "revng/Model/Register.h"

/* TUPLE-TREE-YAML
name: ABIDefinition
doc: Describes a single ABI to be supported by the function within the model
type: struct
fields:
  - name: Architecture
    type: model::Architecture::Values
  - name: Name
    type: std::string
  - name: ArgumentsArePositionBased
    type: bool
    doc: >
      States whether the ABI is focused on preserving the argument order.

      Here's an example function:
      ```
      struct Big; // Is really big, so it can only be passed in memory.
      void function(Big, signed, float, unsigned);
      ```

      If the ABI is position based, the arguments would be passed in
      - memory for the `Big` argument (and a pointer to it in the first GPR).
      - the second GPR register for the `signed` argument.
      - the third vector register for the `float` argument.
      - the forth GPR register for the `unsigned` argument.

      If the ABI is NOT position based, the arguments would be passed in
      - memory for the `Big` argument.
      - the first GPR register for the `signed` argument.
      - the first vector register for the `float` argument.
      - the second GPR register for the `unsigned` argument.

      A typical example of a position-based ABI is `Microsoft_x86_64`,
      a non-position-based one - `SystemV_x86_64`.
  - name: OnlyStartDoubleArgumentsFromAnEvenRegister
    type: bool
    doc: >
      States whether an object that needs two GPRs to fit (5-8 bytes on 32-bit
      architectures and 9-16 bytes on 64-bit systems) is only allowed to start
      from a register with an even index.

      Here's an example function:
      ```
      void function(uint32_t, uint64_t);
      ```

      On a system with 32-bit wide GPRs, the first argument (`uint32_t`) is
      passed using the first allowed GPR (say `r0`).

      The second argument (`uint64_t`) requires two register to fit, so it's
      passed using second and third registers if the ABI allows starting double
      arguments from any register (`r1` and `r2` in this example), or third and
      forth if it only allows starting them from even registers (`r2` and `r3`
      in this example, since `r1` is considered an odd register (the count
      starts from 0, much like C array indexing).

      \note: this option is only applicable for non-position based ABIs
      (if `ArgumentsArePositionBased` is `false`).
  - name: ArgumentsCanBeSplitBetweenRegistersAndStack
    type: bool
    doc: >
      States whether the ABI allows a single object that wouldn't fit into
      a single GPR (9+ bytes on 32-bit systems and 17+ bytes on 64-bit ones)
      to be partially passed in registers with the remainder placed on the stack
      if there are not enough registers to fit the entirety of it.

      As an example, let's say that there is a big object of type `Big` such
      that `sizeof(Big)` is equal to 16 bytes. On 32-bit system it would mean
      having to use four GPRs (`16 == 4 * 4`) to fit it.

      Let's look at an ABI that allocates four registers for passing function
      arguments (`r0-r3`). Then, for a function like
      ```
      void function(uint32_t, Big);
      ```
      the `uint32_t` argument would be passed in the first GPR (`r0`).
      But that would also mean that the remaining three available GPRs are not
      enough to fit the entirety of the `Big` object, meaning it needs to
      either be split between the registers and the memory, or passed using
      the stack. That's exactly what this option states.

      \note: this option is only applicable for non-position based ABIs
      (if `ArgumentsArePositionBased` is `false`).
  - name: UsePointerToCopyForStackArguments
    type: bool
    doc: >
      States how the stack arguments are passed.
      If `UsePointerToCopyForStackArguments` is true, pointers-to-copy are used,
      otherwise - the whole argument is copied onto the stack.

      \note: this only affects the arguments with size exceeding the size of
      a single stack "slot" (which is equal to the GPR size for the architecture
      in question).
  - name: MaximumGPRsPerAggregateArgument
    type: size_t
    doc: >
      States the maximum number of GPRs available to pass a single aggregate
      (a struct, a union, etc.) argument, meaning that it can only be passed in
      the GPRs if `MaximumGPRsPerAggregateArgument` is less than or equal to
      the number of the registers required to fit the object including padding.

      \note: If `MaximumGPRsPerAggregateArgument` is equal to 0, it means that
      the ABI does not allow aggregate arguments to use GPRs.

      \note: If an argument doesn't fit into the specified registers or
      uses irregular padding, the registers are not used and the object is
      passed using the memory (stack, pointer-to-copy, etc.).
  - name: MaximumGPRsPerAggregateReturnValue
    type: size_t
    doc: >
      States the maximum number of GPRs available to return a single aggregate
      (a struct, a union, etc.) value, meaning that it can only be returned
      in the GPRs if `MaximumGPRsPerAggregateReturnValue` is less than or equal
      to the number of the registers required to fit the object including
      padding.

      \note: If `MaximumGPRsPerAggregateReturnValue` is equal to 0, it means
      that the ABI does not allow aggregate return values to use GPRs.

      \note: If a return value doesn't fit into the specified registers or
      uses irregular padding, the registers are not used and the object is
      passed using the memory (stack, pointer-to-copy, etc.).
  - name: MaximumGPRsPerScalarArgument
    type: size_t
    doc: >
      States the maximum number of GPRs available to pass a single scalar
      (`int`, `__int128`, pointer, etc.) argument, meaning that it can only be
      passed in the GPRs if `MaximumGPRsPerScalarArgument` is less than or
      equal to the number of the registers required to fit the object.

      \note: If `MaximumGPRsPerScalarArgument` is equal to 0, it means that
      the ABI does not allow scalar arguments to use GPRs.

      \note: If an argument doesn't fit into the specified registers or
      uses irregular padding, the registers are not used and the object is
      passed using the memory (stack, pointer-to-copy, etc.).
  - name: MaximumGPRsPerScalarReturnValue
    type: size_t
    doc: >
      States the maximum number of GPRs available to return a single scalar
      (`int`, `__int128`, pointer, etc.) value, meaning that it can only be
      returned in the GPRs if `MaximumGPRsPerScalarReturnValue` is less than
      or equal to the number of the registers required to fit the object
      including padding.

      \note: If `MaximumGPRsPerScalarReturnValue` is equal to 0, it means
      that the ABI does not allow scalar return values to use GPRs.

      \note: If a return value doesn't fit into the specified registers or
      uses irregular padding, the registers are not used and the object is
      passed using the memory (stack, pointer-to-copy, etc.).
  - name: GeneralPurposeArgumentRegisters
    sequence:
      type: std::vector
      elementType: model::Register::Values
    doc: >
      Stores the list of general purpose registers allowed to be used for
      passing arguments and the order they are to be used in.
  - name: GeneralPurposeReturnValueRegisters
    sequence:
      type: std::vector
      elementType: model::Register::Values
    doc: >
      Stores the list of general purpose registers allowed to be used for
      returning values and the order they are to be used in.
  - name: VectorArgumentRegisters
    sequence:
      type: std::vector
      elementType: model::Register::Values
    doc: >
      Stores the list of vector registers allowed to be used for passing
      arguments and the order they are to be used in.
  - name: VectorReturnValueRegisters
    sequence:
      type: std::vector
      elementType: model::Register::Values
    doc: >
      Stores the list of vector registers allowed to be used for returning
      values and the order they are to be used in.
  - name: CalleeSavedRegisters
    sequence:
      type: std::vector
      elementType: model::Register::Values
    doc: >
      Stores the list of registers for which the ABI requires the callee to
      preserve the value, meaning that when the callee returns, the value of
      those registers must be the same as it was when the function was called.
  - name: ReturnValueLocationRegister
    type: model::Register::Values
    doc: >
      Specifies a register to be used for returning (or even passing,
      depending on ABI) the pointer to the memory used for returning
      copies of big aggregate objects.

      Can be `model::Register::Invalid` for ABIs that do not support returning
      values by 'pointer-to-copy'.
  - name: CalleeIsResponsibleForStackCleanup
    type: bool
    doc: >
      Specifies who is responsible for cleaning the stack after the function
      call. If equal to `true`, it's the callee, otherwise it the caller.
  - name: StackAlignment
    type: size_t
    doc: >
      States the required alignment of the stack at the point of a function
      call in bytes.

      \note: states minimum value for ABIs supporting multiple different stack
      alignment values, for example, if the ABI requires the stack to be aligned
      on 4 bytes for internal calls but on 8 bytes for interfaces (like 32-bit
      ARM ABI), the value of `StackAlignment` should be equal to 4.
  - name: MinimumStackArgumentSize
    type: size_t
    doc: States the required size for a single stack argument in bytes.
key:
  - Architecture
  - Name
TUPLE-TREE-YAML */

#include "revng/Model/Generated/Early/ABIDefinition.h"

class model::ABIDefinition : public model::generated::ABIDefinition {
public:
  using generated::ABIDefinition::ABIDefinition;

public:
  // TODO: `verify`, `dump` and friends.
};

#include "revng/Model/Generated/Late/ABIDefinition.h"
