#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <concepts>
#include <ranges>
#include <type_traits>

#include "revng/Model/ABI/Definition.h"
#include "revng/Model/ABI/FloatingPointStrategy.h"
#include "revng/Model/ArrayType.h"
#include "revng/Model/DefinedType.h"
#include "revng/Model/EnumDefinition.h"
#include "revng/Model/PointerType.h"
#include "revng/Model/PrimitiveType.h"
#include "revng/Model/Register.h"
#include "revng/Model/StructDefinition.h"
#include "revng/Model/TypedefDefinition.h"
#include "revng/Model/UnionDefinition.h"

namespace abi::FunctionType {

/// The kind of floating point support a value requires from an ABI.
enum class FloatingPointKind {
  NonFloat,
  FloatPrimitive,
  FloatContainingAggregate,
};

/// Determines the kind of floating point support \p Type requires.
///
/// \note this is not perfect, as primitives can be of `Generic` kind -
///       which is impossible to classify.
template<model::AnyType AnyType>
inline FloatingPointKind floatingPointKind(const AnyType &Type) {
  using T = std::remove_cvref_t<AnyType>;

  if constexpr (std::derived_from<T, model::TypeDefinition>) {
    const model::TypeDefinition &Definition = Type;

    if (const auto *S = llvm::dyn_cast<model::StructDefinition>(&Definition)) {
      for (const model::StructField &Field : S->Fields())
        if (floatingPointKind(*Field.Type()) != FloatingPointKind::NonFloat)
          return FloatingPointKind::FloatContainingAggregate;
      return FloatingPointKind::NonFloat;
    }

    if (const auto *U = llvm::dyn_cast<model::UnionDefinition>(&Definition)) {
      for (const model::UnionField &Field : U->Fields())
        if (floatingPointKind(*Field.Type()) != FloatingPointKind::NonFloat)
          return FloatingPointKind::FloatContainingAggregate;
      return FloatingPointKind::NonFloat;
    }

    if (const auto *TD = llvm::dyn_cast<model::TypedefDefinition>(&Definition))
      return floatingPointKind(*TD->UnderlyingType());
    if (const auto *E = llvm::dyn_cast<model::EnumDefinition>(&Definition))
      return floatingPointKind(*E->UnderlyingType());

    return FloatingPointKind::NonFloat;

  } else if constexpr (std::derived_from<T, model::Type>) {
    const model::Type &MT = Type;

    if (const auto *Array = llvm::dyn_cast<model::ArrayType>(&MT))
      return floatingPointKind(*Array->ElementType());

    if (llvm::isa<model::PointerType>(&MT))
      return FloatingPointKind::NonFloat;

    if (const auto *P = llvm::dyn_cast<model::PrimitiveType>(&MT)) {
      if (P->PrimitiveKind() == model::PrimitiveKind::Float)
        return FloatingPointKind::FloatPrimitive;
      return FloatingPointKind::NonFloat;
    }

    if (const auto *D = llvm::dyn_cast<model::DefinedType>(&MT))
      return floatingPointKind(D->unwrap());

    return FloatingPointKind::NonFloat;

  } else {
    static_assert(type_always_false_v<T>, "Unsupported type.");
  }
}

/// Determines what \p ABI does with an *argument* that requires \p Kind of
/// floating point support.
///
/// \note `NonFloat` arguments are not governed by the strategy at all: the
///       regular general purpose register logic takes over for them.
inline FloatingPointStrategy::Values
argumentStrategy(const abi::Definition &ABI, FloatingPointKind Kind) {
  if (Kind == FloatingPointKind::NonFloat)
    return FloatingPointStrategy::NextRegisterFromTheGeneralPurposeRegisterSet;

  return ABI.FloatingPointArgumentStrategy();
}

/// Determines what \p ABI does with a *return value* that requires \p Kind of
/// floating point support.
///
/// \note see `argumentStrategy`.
inline FloatingPointStrategy::Values
returnValueStrategy(const abi::Definition &ABI, FloatingPointKind Kind) {
  if (Kind == FloatingPointKind::NonFloat)
    return FloatingPointStrategy::NextRegisterFromTheGeneralPurposeRegisterSet;

  return ABI.FloatingPointReturnValueStrategy();
}

} // namespace abi::FunctionType
