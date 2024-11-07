#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <array>
#include <cstdint>
#include <span>
#include <string_view>

#include "llvm/Support/YAMLTraits.h"

#include "revng/ADT/Concepts.h"

/// Trait to provide name of the tuple-like class and its fields
template<typename T>
struct TupleLikeTraits {
  enum class Fields {
  };
};

template<typename T>
  requires(std::is_const_v<T>)
struct TupleLikeTraits<T> : TupleLikeTraits<std::remove_const_t<T>> {};

namespace detail {

template<typename T>
using Tuple = typename TupleLikeTraits<T>::tuple;

template<typename T>
concept HasTuple = StrictSpecializationOf<Tuple<T>, std::tuple>;

} // namespace detail

template<typename T>
concept TraitedTupleLike = requires {
  { TupleLikeTraits<T>::Name } -> std::convertible_to<llvm::StringRef>;
  { TupleLikeTraits<T>::FullName } -> std::convertible_to<llvm::StringRef>;
  {
    TupleLikeTraits<T>::FieldNames
  } -> std::convertible_to<const std::span<const llvm::StringRef>>;

  typename TupleLikeTraits<T>::tuple;
  typename TupleLikeTraits<T>::Fields;
} && detail::HasTuple<T> && std::is_enum_v<typename TupleLikeTraits<T>::Fields>;

//
// Implementation of MappingTraits for TupleLikeTraits implementers
//

/// Tuple-like can implement llvm::yaml::MappingTraits inheriting this class
template<TraitedTupleLike T, typename TupleLikeTraits<T>::Fields... Optionals>
struct TupleLikeMappingTraits {
  using Fields = typename TupleLikeTraits<T>::Fields;

  template<Fields Index, size_t I = 0>
  static constexpr bool isOptional() {
    constexpr size_t Count = sizeof...(Optionals);
    constexpr std::array<Fields, Count> OptionalsArray{ Optionals... };
    if constexpr (I < Count) {
      return (OptionalsArray[I] == Index) || isOptional<Index, I + 1>();
    } else {
      return false;
    }
  }

  // Recursive step
  template<size_t I = 0>
  static void mapping(llvm::yaml::IO &IO, T &Obj) {
    if constexpr (I < std::tuple_size_v<T>) {
      auto Name = TupleLikeTraits<T>::FieldNames[I];
      constexpr Fields Field = static_cast<Fields>(I);

      auto &Element = get<I>(Obj);
      if constexpr (isOptional<Field>())
        IO.mapOptional(Name.data(), Element, std::tuple_element_t<I, T>{});
      else
        IO.mapRequired(Name.data(), Element);

      // Recur
      mapping<I + 1>(IO, Obj);
    }
  }
};

template<size_t Index, TraitedTupleLike T>
struct std::tuple_element<Index, T> {
  using type = std::tuple_element_t<Index, typename TupleLikeTraits<T>::tuple>;
};

template<typename T>
using TupleLikeTraitsTuple = typename TupleLikeTraits<T>::tuple;

template<TraitedTupleLike T>
struct std::tuple_size<T>
  : std::integral_constant<size_t, std::tuple_size_v<TupleLikeTraitsTuple<T>>> {
};
