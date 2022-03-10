#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <array>
#include <type_traits>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"

#include "revng/Support/Concepts.h"
#include "revng/Support/Debug.h"

//
// is_specialization
//
template<typename Test, template<typename...> class Ref>
struct is_specialization : std::false_type {};

template<template<typename...> class Ref, typename... Args>
struct is_specialization<Ref<Args...>, Ref> : std::true_type {};

template<template<typename...> class Ref, typename... Args>
struct is_specialization<const Ref<Args...>, Ref> : std::true_type {};

template<typename Test, template<typename...> class Ref>
constexpr bool is_specialization_v = is_specialization<Test, Ref>::value;

static_assert(is_specialization_v<std::vector<int>, std::vector>);
static_assert(is_specialization_v<const std::vector<int>, std::vector>);
static_assert(is_specialization_v<std::pair<int, long>, std::pair>);

//
// always_true and always_false
//
// Since an assert in the `else` branch of an `if_constexpr` condition said
// branch gets instantiated if it doesn't depend on a template, these provide
// an easy way to "fake" dependence on an arbitrary template parameter.
//

template<typename T>
struct type_always_false {
  constexpr static bool value = false;
};
template<typename T>
constexpr inline bool type_always_false_v = type_always_false<T>::value;

template<auto V>
struct value_always_false {
  constexpr static bool value = false;
};
template<auto V>
constexpr inline bool value_always_false_v = value_always_false<V>::value;

template<typename T>
struct type_always_true {
  constexpr static bool value = false;
};
template<typename T>
constexpr inline bool type_always_true_v = type_always_true<T>::value;

template<auto V>
struct value_always_true {
  constexpr static bool value = false;
};
template<auto V>
constexpr inline bool value_always_true_v = value_always_true<V>::value;

//
// HasTupleSize
//

template<class T>
concept HasTupleSize = requires {
  typename std::tuple_size<T>::type;
  { std::tuple_size_v<T> } -> convertible_to<size_t>;
};

static_assert(HasTupleSize<std::tuple<>>);
static_assert(!HasTupleSize<std::vector<int>>);
static_assert(!HasTupleSize<int>);

//
// IsTupleLike
//

namespace revng::detail {

template<class T, std::size_t N>
concept HasTupleElement = requires(T Value) {
  typename std::tuple_element_t<N, std::remove_const_t<T>>;
  { get<N>(Value) } -> convertible_to<std::tuple_element_t<N, T> &>;
};

template<typename T, size_t... N>
constexpr auto checkTupleElementTypes(std::index_sequence<N...>) {
  return (HasTupleElement<T, N> && ...);
}

template<HasTupleSize T>
constexpr auto checkAllTupleElementTypes() {
  auto Sequence = std::make_index_sequence<std::tuple_size_v<T>>();
  return checkTupleElementTypes<T>(Sequence);
}

} // namespace revng::detail

// clang-format off
template<class T>
concept IsTupleLike = (not std::is_reference_v<T>
                       and HasTupleSize<T>
                       and revng::detail::checkAllTupleElementTypes<T>());
// clang-format on

static_assert(IsTupleLike<std::tuple<>>);
static_assert(IsTupleLike<std::tuple<int, int, long>>);
static_assert(IsTupleLike<std::pair<int, int>>);
static_assert(IsTupleLike<std::array<int, 0>>);
static_assert(not IsTupleLike<int>);

//===----------------------------------------------------------------------===//
//     Extra additions to <iterator>
//===----------------------------------------------------------------------===//

namespace revng {
namespace revng::detail {

template<typename FuncTy, typename ItTy>
using ReturnType = decltype(std::declval<FuncTy>()(*std::declval<ItTy>()));

template<typename ItTy,
         typename FuncTy,
         typename FuncReturnTy = ReturnType<FuncTy, ItTy>>
class ProxyMappedIteratorImpl : public llvm::mapped_iterator<ItTy, FuncTy> {
  struct IteratorProxy {
    IteratorProxy(FuncReturnTy &&Value) : Temporary(std::move(Value)) {}
    FuncReturnTy *const operator->() { return &Temporary; }
    FuncReturnTy const *const operator->() const { return &Temporary; }

  private:
    FuncReturnTy Temporary;
  };

public:
  using llvm::mapped_iterator<ItTy, FuncTy>::mapped_iterator;
  using reference = std::decay_t<FuncReturnTy>;

  IteratorProxy operator->() {
    return llvm::mapped_iterator<ItTy, FuncTy>::operator*();
  }
  IteratorProxy const operator->() const {
    return llvm::mapped_iterator<ItTy, FuncTy>::operator*();
  }
};

template<typename ItTy, typename FuncTy>
using ItImpl = std::conditional_t<std::is_object_v<ReturnType<FuncTy, ItTy>>,
                                  ProxyMappedIteratorImpl<ItTy, FuncTy>,
                                  llvm::mapped_iterator<ItTy, FuncTy>>;

} // namespace revng::detail

/// `revng::mapped_iterator` is a specialized version of
/// `llvm::mapped_iterator`.
///
/// It can act as an in-place replacement since it doesn't change the behavior
/// in most cases. The main difference is the fact that when the iterator uses
/// a temporary as a way of remembering its position its lifetime is
/// explicitly prolonged to prevent it from being deleted prematurely (like
/// inside the `operator->` call).
template<typename ItTy, typename FuncTy>
using mapped_iterator = revng::detail::ItImpl<ItTy, FuncTy>;

// `map_iterator` - Provide a convenient way to create `mapped_iterator`s,
// just like `make_pair` is useful for creating pairs...
template<class ItTy, class FuncTy>
inline auto map_iterator(ItTy I, FuncTy F) {
  return mapped_iterator<ItTy, FuncTy>(std::move(I), std::move(F));
};

template<class ContainerTy, class FuncTy>
auto map_range(ContainerTy &&C, FuncTy F) {
  return llvm::make_range(map_iterator(C.begin(), F), map_iterator(C.end(), F));
}

auto dereferenceIterator(auto Iter) {
  return llvm::map_iterator(Iter, [](const auto &Ptr) -> decltype(*Ptr) & {
    return *Ptr;
  });
}

namespace revng::detail {
template<typename T>
using DIT = decltype(dereferenceIterator(std::declval<T>()));
}

template<typename T>
using DereferenceIteratorType = revng::detail::DIT<T>;

auto dereferenceRange(auto &&Range) {
  return llvm::make_range(dereferenceIterator(Range.begin()),
                          dereferenceIterator(Range.end()));
}

template<typename Iterator>
auto mapToValueIterator(Iterator It) {
  const auto GetSecond = [](auto &Pair) -> auto & { return Pair.second; };
  return llvm::map_iterator(It, GetSecond);
}

template<typename T>
using MapToValueIteratorType = decltype(mapToValueIterator(std::declval<T>()));

} // namespace revng

template<typename C>
inline auto skip(unsigned ToSkip, C &&Container)
  -> llvm::iterator_range<decltype(Container.begin())> {
  auto Begin = std::begin(Container);
  while (ToSkip-- > 0)
    Begin++;
  return llvm::make_range(Begin, std::end(Container));
}

//
// slice
//

/// Copy into a std::array a slice of an llvm::ArrayRef
template<size_t Start, size_t Size, typename T>
std::array<T, Size> slice(llvm::ArrayRef<T> Old) {
  std::array<T, Size> Result;
  auto StartIt = Old.begin() + Start;
  std::copy(StartIt, StartIt + Size, Result.begin());
  return Result;
}

/// Copy into a std::array a slice of a std::array
template<size_t Start, size_t Size, typename T, size_t OldSize>
std::array<T, Size> slice(const std::array<T, OldSize> &Old) {
  std::array<T, Size> Result;
  auto StartIt = Old.begin() + Start;
  std::copy(StartIt, StartIt + Size, Result.begin());
  return Result;
}

/// \brief Simple helper function asserting a pointer is not a `nullptr`
template<typename T>
inline T *notNull(T *Pointer) {
  revng_assert(Pointer != nullptr);
  return Pointer;
}
