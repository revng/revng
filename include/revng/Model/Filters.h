#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/ADT/RecursiveCoroutine.h"
#include "revng/ADT/UpcastablePointer.h"
#include "revng/Model/CommonTypeMethods.h"
#include "revng/Model/Type.h"
#include "revng/Model/TypeDefinition.h"

//
// This file provides sample filters to use with containers of model types,
// for example, to iterate all the scalar arguments of a function, you could use
//
// ```cpp
// for (auto &Scalar : Function.Arguments() | model::filter::Scalar)
//   do_stuff(Scalar);
// ```
//
// TODO: add new filters here as you find use cases for them.

namespace model::filter {

//
// Implementation helpers
//

namespace detail {

template<typename Type>
concept HasTypeAccessor = requires(Type &&Value) {
  { Value.Type() };
};

template<typename Type, typename CallableType>
auto unwrap(const Type &Value, CallableType &&Callable) {
  if constexpr (SpecializationOf<Type, UpcastablePointer>) {
    revng_assert(!Value.empty());
    return unwrap(*Value, std::forward<CallableType>(Callable));
  } else if constexpr (SpecializationOf<Type, CommonTypeMethods>) {
    return Callable(Value);
  } else if constexpr (HasTypeAccessor<Type>) {
    return unwrap(Value.Type(), std::forward<CallableType>(Callable));
  } else {
    static_assert(type_always_false_v<Type>);
  }
}

} // namespace detail

//
// Generic filters
//

constexpr auto
  DereferencePointers = std::views::filter([](auto *Ptr) {
                          return Ptr != nullptr;
                        })
                        | std::views::transform([](auto *T) -> decltype(auto) {
                            return *T;
                          });

//
// Type level filters
//

constexpr auto Scalar = std::views::filter([](const auto &Value) {
  return detail::unwrap(Value, [](auto &&V) { return V.isScalar(); });
});
constexpr auto NonScalar = std::views::filter([](const auto &Value) {
  return detail::unwrap(Value, [](auto &&V) { return not V.isScalar(); });
});

//
// Type definition level filters
//

constexpr auto Struct = std::views::transform([](auto &&T) {
                          return llvm::dyn_cast<model::StructDefinition>(&*T);
                        })
                        | DereferencePointers;
constexpr auto Union = std::views::transform([](auto &&T) {
                         return llvm::dyn_cast<model::UnionDefinition>(&*T);
                       })
                       | DereferencePointers;

} // namespace model::filter
