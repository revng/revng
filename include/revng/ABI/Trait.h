#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <concepts>

#include "llvm/ADT/ArrayRef.h"

#include "revng/ABI/AArch64/AAPCS64.h"
#include "revng/ABI/ARM/AAPCS.h"
#include "revng/ABI/MIPS/SystemV_MIPS_o32.h"
#include "revng/ABI/x86/Microsoft_x86_cdecl.h"
#include "revng/ABI/x86/Microsoft_x86_clrcall.h"
#include "revng/ABI/x86/Microsoft_x86_fastcall.h"
#include "revng/ABI/x86/Microsoft_x86_stdcall.h"
#include "revng/ABI/x86/Microsoft_x86_thiscall.h"
#include "revng/ABI/x86/Microsoft_x86_vectorcall.h"
#include "revng/ABI/x86/Pascal_x86.h"
#include "revng/ABI/x86/SystemV_x86.h"
#include "revng/ABI/x86/SystemV_x86_regparm_1.h"
#include "revng/ABI/x86/SystemV_x86_regparm_2.h"
#include "revng/ABI/x86/SystemV_x86_regparm_3.h"
#include "revng/ABI/x86_64/Microsoft_x86_64.h"
#include "revng/ABI/x86_64/Microsoft_x86_64_clrcall.h"
#include "revng/ABI/x86_64/Microsoft_x86_64_vectorcall.h"
#include "revng/ABI/x86_64/SystemV_x86_64.h"
#include "revng/Model/ABI.h"
#include "revng/Model/Register.h"
#include "revng/Support/Concepts.h"

namespace abi {

/// A trait struct used for describing the specifics of an ABI to be used for
/// `CABIFunctionType` representation within the revng model.
///
/// When registering a new ABI, add a specialization of this trait.
///
/// \note At some point these parameters will be moved to the runtime,
/// if possible. That should make adding support for additional ABIs easier.
template<model::ABI::Values V>
struct Trait;

// clang-format off

/// The concept used to check whether the trait struct satisfies
/// the requirements for an ABI defining calling convention trait.
template<template<model::ABI::Values> typename TraitType,
         model::ABI::Values ABI>
concept IsTrait = requires(TraitType<ABI> Trait) {

  /// Indicates the ABI this `Trait` specialization describes.
  { Trait.ABI } -> convertible_to<model::ABI::Values>;
  requires TraitType<ABI>::ABI == ABI;

  /// States whether the ABI is focused on preserving the argument order.
  ///
  /// Here's an example function:
  /// ```
  /// struct Big; // Is really big, so it can only be passed in memory.
  /// void function(Big, signed, float, unsigned);
  /// ```
  ///
  /// If the ABI is position based, the arguments would be passed in
  /// - memory for the `Big` argument (and a pointer to it in the first GPR).
  /// - the second GPR register for the `signed` argument.
  /// - the third vector register for the `float` argument.
  /// - the forth GPR register for the `unsigned` argument.
  ///
  /// If the ABI is NOT position based, the arguments would be passed in
  /// - memory for the `Big` argument.
  /// - the first GPR register for the `signed` argument.
  /// - the first vector register for the `float` argument.
  /// - the second GPR register for the `unsigned` argument.
  ///
  /// A typical example of a position-based ABI is `Microsoft_x86_64`,
  /// a non-position-based one - `SystemV_x86_64`.
  { Trait.ArgumentsArePositionBased } -> convertible_to<bool>;

  /// States whether an object that needs two GPRs to fit (5-8 bytes on 32-bit
  /// architectures and 9-16 bytes on 64-bit systems) is only allowed to start
  /// from a register with an even index.
  ///
  /// Here's an example function:
  /// ```
  /// void function(uint32_t, uint64_t);
  /// ```
  /// On a system with 32-bit wide GPRs, the first argument (`uint32_t`) is
  /// passed using the first allowed GPR (say `r0`).
  ///
  /// The second argument (`uint64_t`) requires two register to fit, so it's
  /// passed using second and third registers if the ABI allows starting double
  /// arguments from any register (`r1` and `r2` in this example), or third and
  /// forth if it only allows starting them from even registers (`r2` and `r3`
  /// in this example, since `r1` is considered an odd register (the count
  /// starts from 0, much like C array indexing).
  ///
  /// \note: this option is only applicable for non-position based ABIs
  /// (if `ArgumentsArePositionBased` is `false`).
  { Trait.OnlyStartDoubleArgumentsFromAnEvenRegister } -> convertible_to<bool>;

  /// States whether the ABI allows a single object that wouldn't fit into
  /// a single GPR (9+ bytes on 32-bit systems and 17+ bytes on 64-bit ones)
  /// to be partially passed in registers with the remainder placed on the stack
  /// if there are not enough registers to fit the entirety of it.
  ///
  /// As an example, let's say that there is a big object of type `Big` such
  /// that `sizeof(Big)` is equal to 16 bytes. On 32-bit system it would mean
  /// having to use four GPRs (`16 == 4 * 4`) to fit it.
  ///
  /// Let's look at an ABI that allocates four registers for passing function
  /// arguments (`r0-r3`). Then, for a function like
  /// ```
  /// void function(uint32_t, Big);
  /// ```
  /// the `uint32_t` argument would be passed in the first GPR (`r0`).
  /// But that would also mean that the remaining three available GPRs are not
  /// enough to fit the entirety of the `Big` object, meaning it needs to
  /// either be split between the registers and the memory, or passed using
  /// the stack. That's exactly what this option states.
  ///
  /// \note: this option is only applicable for non-position based ABIs
  /// (if `ArgumentsArePositionBased` is `false`).
  { Trait.ArgumentsCanBeSplitBetweenRegistersAndStack } -> convertible_to<bool>;

  /// States how the stack arguments are passed.
  /// If `UsePointerToCopyForStackArguments` is true, pointers-to-copy are used,
  /// otherwise - the whole argument is copied onto the stack.
  ///
  /// \note: this only affects the arguments with size exceeding the size of
  /// a single stack "slot" (which is equal to the GPR size for the architecture
  /// in question).
  { Trait.UsePointerToCopyForStackArguments } -> convertible_to<bool>;

  /// States the maximum number of GPRs available to pass a single aggregate
  /// (a struct, a union, etc.) argument, meaning that it can only be passed in
  /// the GPRs if `MaximumGPRsPerAggregateArgument` is less than or equal to
  /// the number of the registers required to fit the object including padding.
  ///
  /// \note: If `MaximumGPRsPerAggregateArgument` is equal to 0, it means that
  /// the ABI does not allow aggregate arguments to use GPRs.
  ///
  /// \note: If an argument doesn't fit into the specified registers or
  /// uses irregular padding, the registers are not used and the object is
  /// passed using the memory (stack, pointer-to-copy, etc.).
  { Trait.MaximumGPRsPerAggregateArgument } -> convertible_to<size_t>;

  /// States the maximum number of GPRs available to return a single aggregate
  /// (a struct, a union, etc.) value, meaning that it can only be returned
  /// in the GPRs if `MaximumGPRsPerAggregateReturnValue` is less than or equal
  /// to the number of the registers required to fit the object including
  /// padding.
  ///
  /// \note: If `MaximumGPRsPerAggregateReturnValue` is equal to 0, it means
  /// that the ABI does not allow aggregate return values to use GPRs.
  ///
  /// \note: If a return value doesn't fit into the specified registers or
  /// uses irregular padding, the registers are not used and the object is
  /// passed using the memory (stack, pointer-to-copy, etc.).
  { Trait.MaximumGPRsPerAggregateReturnValue } -> convertible_to<size_t>;

  /// States the maximum number of GPRs available to pass a single scalar
  /// (`int`, `__int128`, pointer, etc.) argument, meaning that it can only be
  /// passed in the GPRs if `MaximumGPRsPerScalarArgument` is less than or
  /// equal to the number of the registers required to fit the object.
  ///
  /// \note: If `MaximumGPRsPerScalarArgument` is equal to 0, it means that
  /// the ABI does not allow scalar arguments to use GPRs.
  ///
  /// \note: If an argument doesn't fit into the specified registers or
  /// uses irregular padding, the registers are not used and the object is
  /// passed using the memory (stack, pointer-to-copy, etc.).
  { Trait.MaximumGPRsPerScalarArgument } -> convertible_to<size_t>;

  /// States the maximum number of GPRs available to return a single scalar
  /// (`int`, `__int128`, pointer, etc.) value, meaning that it can only be
  /// returned in the GPRs if `MaximumGPRsPerScalarReturnValue` is less than
  /// or equal to the number of the registers required to fit the object
  /// including padding.
  ///
  /// \note: If `MaximumGPRsPerScalarReturnValue` is equal to 0, it means
  /// that the ABI does not allow scalar return values to use GPRs.
  ///
  /// \note: If a return value doesn't fit into the specified registers or
  /// uses irregular padding, the registers are not used and the object is
  /// passed using the memory (stack, pointer-to-copy, etc.).
  { Trait.MaximumGPRsPerScalarReturnValue } -> convertible_to<size_t>;

  /// Stores the list of general purpose registers allowed to be used for
  /// passing arguments and the order they are to be used in.
  { Trait.GeneralPurposeArgumentRegisters } ->
    convertible_to<llvm::ArrayRef<model::Register::Values>>;

  /// Stores the list of general purpose registers allowed to be used for
  /// returning values and the order they are to be used in.
  { Trait.GeneralPurposeReturnValueRegisters } ->
    convertible_to<llvm::ArrayRef<model::Register::Values>>;

  /// Stores the list of vector registers allowed to be used for passing
  /// arguments and the order they are to be used in.
  { Trait.VectorArgumentRegisters } ->
    convertible_to<llvm::ArrayRef<model::Register::Values>>;

  /// Stores the list of vector registers allowed to be used for returning
  /// values and the order they are to be used in.
  { Trait.VectorReturnValueRegisters } ->
    convertible_to<llvm::ArrayRef<model::Register::Values>>;

  /// Stores the list of registers for which the ABI requires the callee to
  /// preserve the value, meaning that when the callee returns, the value of
  /// those registers must be the same as it was when the function was called.
  { Trait.CalleeSavedRegisters } ->
    convertible_to<llvm::ArrayRef<model::Register::Values>>;

  /// Specifies a register to be used for returning (or even passing,
  /// depending on ABI) the pointer to the memory used for returning
  /// copies of big aggregate objects.
  ///
  /// Can be `model::Register::Invalid` for ABIs that do not support returning
  /// values by 'pointer-to-copy'.
  { Trait.ReturnValueLocationRegister } ->
    convertible_to<model::Register::Values>;

};

namespace detail {

template<template<model::ABI::Values> typename Trait, size_t StartFrom = 0>
consteval bool verifyTraitSpecializations() {
  if constexpr (StartFrom == model::ABI::Count) {
    return true;
  } else {
    constexpr model::ABI::Values CurrentABI = model::ABI::Values(StartFrom);
    constexpr bool IsValid = IsTrait<Trait, CurrentABI>;
    static_assert(IsValid); // Improves error messages.

    constexpr bool R = verifyTraitSpecializations<Trait, StartFrom + 1>();
    return IsValid && R;
  }
}
static_assert(verifyTraitSpecializations<Trait, 1>());

} // namespace detail

// clang-format on

} // namespace abi
