//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <algorithm>
#include <optional>

#include "llvm/ADT/STLExtras.h"

#include "revng/ADT/RecursiveCoroutine.h"
#include "revng/Model/ABI/Definition.h"
#include "revng/Model/ArrayType.h"
#include "revng/Model/DefinedType.h"
#include "revng/Model/EnumDefinition.h"
#include "revng/Model/PointerType.h"
#include "revng/Model/PrimitiveType.h"
#include "revng/Model/StructDefinition.h"
#include "revng/Model/TypedefDefinition.h"
#include "revng/Model/UnionDefinition.h"

#include "ValueDistributor.h"

// The following is an implementation of the SystemV x86-64 register selection
// logic, guarded behind the `UseSystemVx86_64RegisterSelectionLogic` option of
// `abi::Definition`.
//
// It's an implementation of the section 3.2.3 of the SystemV x86-64 psABI:
// "Parameter Passing".
//
// *Every* class the specification defines is spelled out, including the ones
// the model cannot represent today. They are kept explicit on purpose. We will
// need them all eventually, so not much reason to fragment the algorithm for
// a temporary simplification.
//
// Note that we also introduce a single class the specification never mentions:
// `Unsupported`, as we have the luxury to fail when we don't like the input,
// unlike a compiler.
//
// All the kinds we do not support (like `X87`) are treated as `Unsupported` if
// they ever end up in the output of the classifier.

namespace abi::FunctionType {

namespace {

/// The eightbyte classes defined by the SystemV x86-64 ABI.
enum class Class {
  NoClass,
  Integer,
  SSE,
  SSEUP,
  X87,
  X87UP,
  ComplexX87,
  Memory,

  /// This class is specific to our implementation and serves as a response to
  /// inputs we cannot handle easily.
  Unsupported,
};

/// Merges two classes when they apply to the same eightbyte.
Class combine(Class LHS, Class RHS) {
  if (LHS == RHS)
    return LHS;

  // `Unsupported` takes precedence over anything else.
  if (LHS == Class::Unsupported or RHS == Class::Unsupported)
    return Class::Unsupported;

  // `NoClass` is always overridden.
  if (LHS == Class::NoClass)
    return RHS;
  if (RHS == Class::NoClass)
    return LHS;

  // `Memory` + anything else = `Memory`.
  if (LHS == Class::Memory or RHS == Class::Memory)
    return Class::Memory;

  // `Integer` + anything but `Memory` = `Integer`.
  if (LHS == Class::Integer or RHS == Class::Integer)
    return Class::Integer;

  // `x87` family never shares an eightbyte with anything else.
  constexpr std::array X87Classes = {
    Class::X87,
    Class::X87UP,
    Class::ComplexX87,
  };
  if (llvm::is_contained(X87Classes, LHS)
      or llvm::is_contained(X87Classes, RHS))
    return Class::Memory;

  // Only the `SSE` and `SSEUP` are left - and `SSE` takes precedence.
  return Class::SSE;
}

constexpr uint64_t EightbyteSize = 8;
uint64_t eightbyteCount(uint64_t Size) {
  return (Size + EightbyteSize - 1) / EightbyteSize;
}

/// Combines \p Classes into \p Result, starting at the eightbyte that
/// \p Offset falls in.
void combineFields(uint64_t Offset,
                   llvm::ArrayRef<Class> Classes,
                   llvm::MutableArrayRef<Class> Result) {
  const uint64_t Start = Offset / EightbyteSize;
  revng_assert(Start + Classes.size() <= Result.size());
  for (uint64_t I = 0; I < Classes.size(); ++I)
    Result[Start + I] = combine(Result[Start + I], Classes[I]);
}

llvm::SmallVector<Class, 8> classifyScalar(const model::Type &Type) {
  const uint64_t Size = *Type.size();
  revng_assert(Size > 0);

  const uint64_t Count = eightbyteCount(Size);

  if (const auto *P = llvm::dyn_cast<model::PointerType>(&Type)) {
    revng_assert(Count == 1, "Pointer > 8 bytes?");
    return llvm::SmallVector<Class, 8>(Count, Class::Integer);

  } else if (const auto *P = llvm::dyn_cast<model::PrimitiveType>(&Type)) {
    switch (P->PrimitiveKind()) {
    case model::PrimitiveKind::PointerOrNumber:
    case model::PrimitiveKind::Number:
    case model::PrimitiveKind::Unsigned:
    case model::PrimitiveKind::Signed:
      return llvm::SmallVector<Class, 8>(Count, Class::Integer);

    case model::PrimitiveKind::Float:
      switch (Size) {
      case 4:
      case 8:
        return llvm::SmallVector<Class, 8>{ Class::SSE };

        // TODO: map the other classes:
        // - 2-byte `_Float16`: `SSE`.
        // - 10- and 12-byte `long double`: `X87`+`X87UP`.
        // - 16-byte `__float128`, `_Decimal128` or `__m128`: `SSE`+`SSEUP`.

      default:
        // Mark all the unknown primitives as `Unsupported` for the purposes
        // of this
        return llvm::SmallVector<Class, 8>(Count, Class::Unsupported);
      }

    case model::PrimitiveKind::Generic:
      // Assume the special 32- and 64-byte generics are there because of
      // ymm/zmm and assign them `SSE`+`SSEUP` class.
      if (Size == 32 or Size == 64) {
        llvm::SmallVector<Class, 8> Result(Count, Class::SSEUP);
        Result.front() = Class::SSE;
        return Result;
      }

      // `Integer` is a bad fit for a `Generic`, as it overrides most of
      // the other classes on merge. Consider using something softer, like
      // `NoClass` (used for struct padding).
      return llvm::SmallVector<Class, 8>(Count, Class::Integer);

    case model::PrimitiveKind::Void:
    case model::PrimitiveKind::Invalid:
    case model::PrimitiveKind::Count:
    default:
      revng_abort("Unsupported primitive");
    }
  }

  revng_abort("A scalar that's not a primitive or pointer?");
}

RecursiveCoroutine<llvm::SmallVector<Class, 8>>
classifyType(const model::TypeDefinition &Type);

RecursiveCoroutine<llvm::SmallVector<Class, 8>>
classifyType(const model::Type &Type) {
  if (const auto *D = llvm::dyn_cast<model::DefinedType>(&Type)) {
    rc_return rc_recur classifyType(D->unwrap());

  } else if (const auto *A = llvm::dyn_cast<model::ArrayType>(&Type)) {
    const model::Type &ElementType = *A->ElementType();
    const uint64_t ElementSize = *ElementType.size();
    revng_assert(ElementSize > 0);

    llvm::SmallVector<Class, 8> Result(eightbyteCount(*Type.size()),
                                       Class::NoClass);

    llvm::SmallVector<Class, 8> Element = rc_recur classifyType(ElementType);
    for (uint64_t I = 0; I < A->ElementCount(); ++I)
      combineFields(I * ElementSize, Element, Result);

    rc_return Result;
  } else {
    rc_return classifyScalar(Type);
  }
}

RecursiveCoroutine<llvm::SmallVector<Class, 8>>
classifyType(const model::TypeDefinition &Type) {
  if (const auto *S = llvm::dyn_cast<model::StructDefinition>(&Type)) {
    llvm::SmallVector<Class, 8> Result(eightbyteCount(*Type.size()),
                                       Class::NoClass);

    for (const model::StructField &F : S->Fields())
      combineFields(F.Offset(), rc_recur classifyType(*F.Type()), Result);

    rc_return Result;

  } else if (const auto *U = llvm::dyn_cast<model::UnionDefinition>(&Type)) {
    llvm::SmallVector<Class, 8> Result(eightbyteCount(*Type.size()),
                                       Class::NoClass);

    for (const model::UnionField &Field : U->Fields())
      combineFields(0, rc_recur classifyType(*Field.Type()), Result);

    rc_return Result;

  } else if (const auto *E = llvm::dyn_cast<model::EnumDefinition>(&Type)) {
    rc_return rc_recur classifyType(*E->UnderlyingType());

  } else if (const auto *T = llvm::dyn_cast<model::TypedefDefinition>(&Type)) {
    rc_return rc_recur classifyType(*T->UnderlyingType());

  } else {
    revng_abort("Function type definitions must not reach here.");
  }
}

/// Classifies \p Type as prescribed by the SystemV x86-64 ABI.
///
/// \returns the register class of every eightbyte of \p Type,
///          an empty list if the value has to be passed in memory,
///          and `std::nullopt` if the type is not supported.
template<model::AnyType AnyType>
std::optional<llvm::SmallVector<Class, 8>>
classify(const AnyType &Type, bool IsNaturallyAligned) {
  uint64_t Size = *Type.size();
  revng_assert(Size > 0);

  if (Size > 8 * EightbyteSize or not IsNaturallyAligned) {
    // Really big or unnaturally aligned - straight onto the stack.
    return llvm::SmallVector<Class, 8>{};
  }

  llvm::SmallVector<Class, 8> Result = rc_eval(classifyType(Type));
  if (llvm::is_contained(Result, Class::Unsupported)) {
    // If any of the eightbytes is `Unsupported`, the whole value is unsupported
    return std::nullopt;
  }

  if (llvm::is_contained(Result, Class::Memory)) {
    // If any of the eightbytes is `Memory`, the whole value should go onto
    // the stack.
    return llvm::SmallVector<Class, 8>{};
  }

  for (const auto &[Previous, Next] : zip_pairs(Result)) {
    if (Next == Class::X87UP and Previous != Class::X87) {
      // TODO: this is the big distinction of the Apple x64 ABI flavor:
      //       it should return `SSE` here instead of going straight to
      //       the stack.
      return llvm::SmallVector<Class, 8>{};
    }
  }

  // `SSEUP` not preceded by `SSE` is converted to `SSE`.
  for (auto [Previous, Next] : zip_pairs(Result)) {
    if (Next == Class::SSEUP and Previous != Class::SSE)
      Next = Class::SSE;
  }

  // Count number of non-`UP` eightbytes.
  // If we require more than two, the whole value goes to memory.
  uint64_t NeededRegisters = llvm::count_if(Result, [](Class C) {
    return C != Class::SSEUP and C != Class::X87UP;
  });
  if (NeededRegisters > 2)
    return llvm::SmallVector<Class, 8>{};

  // A gate to prevent temporarily unsupported classes leaking outside
  // this function.
  auto TemporarilyUnsupportedFilter = [](Class C) {
    return llvm::is_contained({ Class::X87, Class::X87UP, Class::ComplexX87 },
                              C);
  };
  if (llvm::count_if(Result, TemporarilyUnsupportedFilter) > 0)
    return std::nullopt;

  return Result;
}

/// The argument helper for the \ref allocate below.
///
/// It greatly improves the readability of the prototype, by tying counts of
/// different register kinds together.
struct RegisterCounts {
  uint64_t GeneralPurposeRegisters = 0;
  uint64_t SSERegisters = 0;
};

/// The return value of the \ref allocate below.
///
/// A simplified version of a register portion list. Namely, only the `Size`
/// field is represented. `Offset` is assumed to always be zero and `Register`
/// is assumed to be filled in by the callee.
struct RegisterSizes {
  llvm::SmallVector<uint64_t, 2> GeneralPurposeRegisters;
  llvm::SmallVector<uint64_t, 2> SSERegisters;
};

/// Decides which registers a value classified as \p Classes occupies, and how
/// many bytes of it each of them carries.
///
/// \param Size the size of the value.
/// \param Used how many registers are already allocated.
/// \param Maximum total register counts per kind.
///
/// \returns the size to allocate each register with, empty for both kinds if
///          the value should end up in memory.
std::optional<RegisterSizes> allocate(llvm::ArrayRef<Class> Classes,
                                      uint64_t Size,
                                      RegisterCounts Used,
                                      RegisterCounts Maximum) {
  RegisterSizes Result;

  uint64_t EightbyteIndex = 0;
  for (Class Current : Classes) {
    const uint64_t Remaining = Size - EightbyteIndex * EightbyteSize;
    ++EightbyteIndex;

    uint64_t CurrentPortionSize = std::min(EightbyteSize, Remaining);
    switch (Current) {
    case Class::Integer:
      // Allocate another general purpose register.
      Result.GeneralPurposeRegisters.push_back(CurrentPortionSize);
      break;

    case Class::SSE:
      // Allocate another SSE register.
      Result.SSERegisters.push_back(CurrentPortionSize);
      break;

    case Class::SSEUP:
      // Extend the previous SSE register.
      revng_assert(not Result.SSERegisters.empty());
      Result.SSERegisters.back() += CurrentPortionSize;
      break;

    case Class::NoClass:
      // This can only happen because of struct padding.
      // There is no way to tell where this eightbyte should be placed,
      // so let's do the same thing we do for generic primitives and use GPRs.
      Result.GeneralPurposeRegisters.push_back(CurrentPortionSize);
      break;

    case Class::X87:
    case Class::X87UP:
    case Class::ComplexX87:
    case Class::Memory:
    case Class::Unsupported:
    default:
      revng_unreachable("These classes should never reach this point.");
    }
  }

  // If we don't have enough registers, the entire value is not allowed to use
  // registers.
  const uint64_t NeededGPRs = Used.GeneralPurposeRegisters
                              + Result.GeneralPurposeRegisters.size();
  const uint64_t NeededSSERegisters = Used.SSERegisters
                                      + Result.SSERegisters.size();
  if (NeededGPRs > Maximum.GeneralPurposeRegisters
      or NeededSSERegisters > Maximum.SSERegisters)
    return std::nullopt;

  return Result;
}

/// A helper struct to make \ref hydratePortions interface easier to read.
struct RegisterPool {
  llvm::ArrayRef<model::Register::Values> Registers;
  uint64_t &Used;
};

/// Turns the \ref Sizes \ref allocate handed out into portions, taking
/// registers from the corresponding pools.
RegisterPortions hydratePortions(const RegisterSizes &Sizes,
                                 RegisterPool GPRs,
                                 RegisterPool FPRs) {
  RegisterPortions Result;

  for (uint64_t Size : Sizes.GeneralPurposeRegisters)
    Result.emplace_back(GPRs.Registers[GPRs.Used++], Size);

  for (uint64_t Size : Sizes.SSERegisters)
    Result.emplace_back(FPRs.Registers[FPRs.Used++], Size);

  return Result;
}

/// Applies the SystemV x86-64 eightbyte classification to \p Type and places
/// it in the registers of \p GPRs and \p SSE.
///
/// \returns a value when it's in registers OR it is unsupported.
///           `std::nullopt` indicates that the stack should be used instead,
///           letting the caller to redirect to the usual distribution logic.
template<model::AnyType AnyType>
std::optional<DistributedValue>
distributeForSystemVx86_64(const abi::Definition &ABI,
                           const AnyType &Type,
                           RegisterPool GPRs,
                           RegisterPool SSE) {
  const uint64_t Size = *Type.size();
  revng_assert(Size > 0);

  abi::Definition::AlignmentCache Cache;
  const bool IsNaturallyAligned = *ABI.hasNaturalAlignment(Type, Cache);
  const std::optional<llvm::SmallVector<Class, 8>>
    Classes = classify(Type, IsNaturallyAligned);

  if (not Classes.has_value()) {
    // Classification hard-failed, meaning we cannot support this value.
    return DistributedValue::ignoredValuePlaceholder(Size);
  }

  if (Classes->empty()) {
    // Empty class list is a short-hand to use stack.
    return std::nullopt;
  }

  std::optional PortionSizes = allocate(*Classes,
                                        Size,
                                        { GPRs.Used, SSE.Used },
                                        { GPRs.Registers.size(),
                                          SSE.Registers.size() });
  if (not PortionSizes.has_value()) {
    // `allocate` returns `std::nullopt` when we do not have enough registers
    // left to house the value, meaning the entire thing should go on the stack.
    return std::nullopt;
  }

  // We have our distribution, hydrate the internal portions into the real thing
  // and return the result.
  DistributedValue Result;
  Result.Size = Size;
  Result.Registers = hydratePortions(*PortionSizes, GPRs, SSE);
  return std::move(Result);
}

} // namespace

DistributedValues ArgumentDistributor::systemV(const model::Type &Type) {
  ++ArgumentIndex;

  namespace FPS = abi::FloatingPointStrategy;
  constexpr auto DedicatedStrategy = FPS::NextRegisterFromTheDedicatedSet;
  if (ABI.FloatingPointArgumentStrategy() != DedicatedStrategy)
    return { DistributedValue::ignoredValuePlaceholder(*Type.size()) };

  RegisterPool GPRs{ ABI.GeneralPurposeArgumentRegisters(),
                     UsedGeneralPurposeRegisterCount };
  RegisterPool SSE{ ABI.FloatArgumentRegisters(), UsedFloatRegisterCount };
  if (auto Value = distributeForSystemVx86_64(ABI, Type, GPRs, SSE))
    return { std::move(*Value) };

  auto &&[Values, _] = distribute(Type, {}, 0, 0, true);
  return std::move(Values);
}

DistributedValues
ArgumentDistributor::systemV(const model::TypeDefinition &Type) {
  ++ArgumentIndex;

  namespace FPS = abi::FloatingPointStrategy;
  constexpr auto DedicatedStrategy = FPS::NextRegisterFromTheDedicatedSet;
  if (ABI.FloatingPointArgumentStrategy() != DedicatedStrategy)
    return { DistributedValue::ignoredValuePlaceholder(*Type.size()) };

  RegisterPool GPRs{ ABI.GeneralPurposeArgumentRegisters(),
                     UsedGeneralPurposeRegisterCount };
  RegisterPool SSE{ ABI.FloatArgumentRegisters(), UsedFloatRegisterCount };
  if (auto Value = distributeForSystemVx86_64(ABI, Type, GPRs, SSE))
    return { std::move(*Value) };

  auto &&[Values, _] = distribute(Type, {}, 0, 0, true);
  return std::move(Values);
}

DistributedValue
ReturnValueDistributor::systemVx86_64ReturnValue(const model::Type &Type) {
  namespace FPS = abi::FloatingPointStrategy;
  constexpr auto DedicatedStrategy = FPS::NextRegisterFromTheDedicatedSet;
  if (ABI.FloatingPointArgumentStrategy() != DedicatedStrategy)
    return { DistributedValue::ignoredValuePlaceholder(*Type.size()) };

  // There must be at most one return value, so no reason to involve
  // the counters of the distributor.
  uint64_t UnusedGPRs = 0;
  uint64_t UnusedSSERegisters = 0;

  RegisterPool GPRs{ ABI.GeneralPurposeReturnValueRegisters(), UnusedGPRs };
  RegisterPool SSE{ ABI.FloatReturnValueRegisters(), UnusedSSERegisters };
  if (auto Value = distributeForSystemVx86_64(ABI, Type, GPRs, SSE))
    return std::move(*Value);

  auto &&[Result, _] = distribute(Type, {}, 0, 0, true);
  revng_assert(Result.size() == 1, "Return values should never be padded.");
  return Result[0];
}

} // namespace abi::FunctionType
