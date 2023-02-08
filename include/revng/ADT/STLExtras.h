#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <array>
#include <optional>
#include <set>
#include <string_view>
#include <type_traits>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/iterator_range.h"

#include "revng/ADT/Concepts.h"
#include "revng/Support/Debug.h"

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

//===----------------------------------------------------------------------===//
//     Extra additions to <iterator>
//===----------------------------------------------------------------------===//

namespace revng {
namespace detail {

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

} // namespace detail

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

namespace detail {
template<typename T>
using DIT = decltype(dereferenceIterator(std::declval<T>()));
}

template<typename T>
using DereferenceIteratorType = revng::detail::DIT<T>;

template<typename T>
using DereferenceRangeType = llvm::iterator_range<revng::detail::DIT<T>>;

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

/// Simple helper function asserting a pointer is not a `nullptr`
template<typename T>
inline T *notNull(T *Pointer) {
  revng_assert(Pointer != nullptr);
  return Pointer;
}

inline llvm::ArrayRef<uint8_t> toArrayRef(llvm::StringRef Data) {
  auto Pointer = reinterpret_cast<const uint8_t *>(Data.data());
  return llvm::makeArrayRef<uint8_t>(Pointer, Data.size());
}

//
// append
//
template<ranges::sized_range FromType, ranges::sized_range ToType>
auto append(FromType &&From, ToType &To) {
  size_t ExistingElementCount = To.size();
  To.resize(ExistingElementCount + From.size());
  return llvm::copy(From, std::next(To.begin(), ExistingElementCount));
}

/// Intersects two std::sets
template<typename T>
std::set<T *> intersect(const std::set<T *> &First, const std::set<T *> &Last) {
  std::set<T *> Output;
  std::set_intersection(First.begin(),
                        First.end(),
                        Last.begin(),
                        Last.end(),
                        std::inserter(Output, Output.begin()));
  return Output;
}

//
// constexpr repeat
//

namespace detail {

template<typename TemplatedCallableType, std::size_t... Indices>
constexpr void constexprRepeatImpl(std::index_sequence<Indices...>,
                                   TemplatedCallableType &&Callable) {
  (Callable.template operator()<Indices>(), ...);
}

template<typename TemplatedCallableType, std::size_t... Indices>
constexpr bool constexprAndImpl(std::index_sequence<Indices...>,
                                TemplatedCallableType &&Callable) {
  return (Callable.template operator()<Indices>() && ...);
}

template<typename TemplatedCallableType, std::size_t... Indices>
constexpr bool constexprOrImpl(std::index_sequence<Indices...>,
                               TemplatedCallableType &&Callable) {
  return (Callable.template operator()<Indices>() || ...);
}

} // namespace detail

template<std::size_t IterationCount, typename CallableType>
constexpr void constexprRepeat(CallableType &&Callable) {
  detail::constexprRepeatImpl(std::make_index_sequence<IterationCount>(),
                              std::forward<CallableType>(Callable));
}

template<std::size_t IterationCount, typename CallableType>
constexpr bool constexprAnd(CallableType &&Callable) {
  return detail::constexprAndImpl(std::make_index_sequence<IterationCount>(),
                                  std::forward<CallableType>(Callable));
}

template<std::size_t IterationCount, typename CallableType>
constexpr bool constexprOr(CallableType &&Callable) {
  return detail::constexprOrImpl(std::make_index_sequence<IterationCount>(),
                                 std::forward<CallableType>(Callable));
}

namespace examples {
using namespace std::string_view_literals;

template<std::size_t Count>
consteval std::size_t fullSize(std::array<std::string_view, Count> Components,
                               std::string_view Separator) {
  std::size_t Result = Separator.size() * Count;
  constexprRepeat<Count>([&Result, &Components]<std::size_t Index> {
    Result += std::get<Index>(Components).size();
  });
  return Result;
}

inline constexpr std::array Components = { "instruction"sv,
                                           "0x401000:Code_x86_64"sv,
                                           "0x402000:Code_x86_64"sv,
                                           "0x403000:Code_x86_64"sv };
static_assert(fullSize(Components, "/"sv) == 75);

} // namespace examples

//
// constexpr split
//

namespace detail {

template<std::size_t N, std::size_t I = 0>
inline constexpr bool
constexprSplitHelper(std::array<std::string_view, N> &Result,
                     std::string_view Separator,
                     std::string_view Input) {
  std::size_t Position = Input.find(Separator);
  if constexpr (I < N - 1) {
    if (Position == std::string_view::npos)
      return false;

    Result[I] = Input.substr(0, Position);
    return constexprSplitHelper<N, I + 1>(Result,
                                          Separator,
                                          Input.substr(Position + 1));
  } else {
    if (Position != std::string_view::npos)
      return false;

    Result[I] = Input;
    return true;
  }
}

} // namespace detail

/// I'm forced to implement my own split because `llvm::StringRef`'s alternative
/// is not `constexpr`-compatible.
///
/// This also uses `std::string_view` instead of `llvm::StringRef` because its
/// `find` member is constexpr - hence at least that member doesn't have to be
/// reimplemented
template<std::size_t N>
inline constexpr std::optional<std::array<std::string_view, N>>
constexprSplit(std::string_view Separator, std::string_view Input) {
  if (std::array<std::string_view, N> Result;
      detail::constexprSplitHelper<N>(Result, Separator, Input))
    return Result;
  else
    return std::nullopt;
}

inline void
replaceAll(std::string &Input, const std::string &From, const std::string &To) {
  if (From.empty())
    return;

  size_t Start = 0;
  while ((Start = Input.find(From, Start)) != std::string::npos) {
    Input.replace(Start, From.length(), To);
    Start += To.length();
  }
}

//
// `constexpr` versions of the llvm algorithm adaptors.
//

namespace revng {

/// \note use `llvm::find` instead after it's made `constexpr`.
template<typename R, typename T>
constexpr decltype(auto) find(R &&Range, const T &Value) {
  return std::find(std::begin(Range), std::end(Range), Value);
}

/// \note use `llvm::is_contained` instead after it's made `constexpr`.
template<typename R, typename T> // NOLINTNEXTLINE
constexpr bool is_contained(R &&Range, const T &Value) {
  return revng::find(Range, Value) != std::end(Range);
}

static_assert(is_contained(std::array{ 1, 2, 3 }, 2) == true);
static_assert(is_contained(std::array{ 1, 2, 3 }, 4) == false);

} // namespace revng
