#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <array>
#include <iterator>
#include <optional>
#include <ranges>
#include <set>
#include <string_view>
#include <type_traits>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/iterator_range.h"

#include "revng/ADT/CompilationTime.h"
#include "revng/ADT/Concepts.h"
#include "revng/Support/Assert.h"
#include "revng/Support/Debug.h"

/// Tuple helpers to be used in type definitions
template<typename T1>
using make_tuple_t = decltype(std::make_tuple(std::declval<T1>()));

template<typename T1, typename T2>
using tuple_cat_t = decltype(std::tuple_cat(std::declval<T1>(),
                                            std::declval<T2>()));

//
// always_true and always_false
//
// Since an assert in the `else` branch of an `if_constexpr` condition said
// branch gets instantiated if it doesn't depend on a template, these provide
// an easy way to "fake" dependence on an arbitrary template parameter.
//
// TODO: these will no longer be necessary after we switch to clang 17+ (defect
//       report 2518), don't forget to replace their usages with just `false`.

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

//
// skip
//

namespace revng::detail {

template<bool SafeMode, typename IteratorType>
inline auto skipImpl(IteratorType &&From,
                     IteratorType &&To,
                     size_t Front = 0,
                     size_t Back = 0) -> llvm::iterator_range<IteratorType> {

  std::ptrdiff_t TotalSkippedCount = Front + Back;
  if constexpr (std::forward_iterator<IteratorType>) {
    // We cannot check on the input iterators because it's going to consume
    // them.

    if (std::distance(From, To) < TotalSkippedCount) {
      if constexpr (SafeMode) {
        revng_abort("Input range has fewer elements than the intended skip.");
      } else {
        // Quietly return an empty range if there are more skips requested than
        // the total number of elements the input range contains.
        return llvm::make_range(To, To);
      }
    }
  }

  std::decay_t<IteratorType> Begin{ From };
  std::advance(Begin, Front);

  std::decay_t<IteratorType> End{ To };
  std::advance(End, -(std::ptrdiff_t) Back);

  return llvm::make_range(std::move(Begin), std::move(End));
}

template<std::bidirectional_iterator T>
inline decltype(auto)
skip(T &&From, T &&To, size_t Front = 0, size_t Back = 0) {
  return skipImpl<true>(std::forward<T>(From),
                        std::forward<T>(To),
                        Front,
                        Back);
}

template<std::input_iterator T>
inline decltype(auto) skip_front(T &&From, T &&To, size_t SkippedCount = 1) {
  return skipImpl<true>(std::forward<T>(From),
                        std::forward<T>(To),
                        SkippedCount,
                        0);
}

template<std::bidirectional_iterator T>
inline decltype(auto) skip_back(T &&From, T &&To, size_t SkippedCount = 1) {
  return skipImpl<true>(std::forward<T>(From),
                        std::forward<T>(To),
                        0,
                        SkippedCount);
}

} // namespace revng::detail

template<std::ranges::range T>
inline decltype(auto) skip(T &&Range, size_t Front = 0, size_t Back = 0) {
  return revng::detail::skip(Range.begin(), Range.end(), Front, Back);
}

template<std::ranges::range T>
inline decltype(auto) skip_front(T &&Range, size_t SkippedCount = 1) {
  return revng::detail::skip_front(Range.begin(), Range.end(), SkippedCount);
}

template<std::ranges::range T>
inline decltype(auto) skip_back(T &&Range, size_t SkippedCount = 1) {
  return revng::detail::skip_back(Range.begin(), Range.end(), SkippedCount);
}

// TODO: reimplement in terms of `std::views::adjacent` once that's available.
template<std::ranges::range T>
inline decltype(auto) zip_pairs(T &&Range) {
  return llvm::zip(revng::detail::skipImpl<false>(Range.begin(),
                                                  Range.end(),
                                                  0,
                                                  1),
                   revng::detail::skipImpl<false>(Range.begin(),
                                                  Range.end(),
                                                  1,
                                                  0));
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
  return llvm::ArrayRef<uint8_t>(Pointer, Data.size());
}

template<typename T, typename ValueType>
concept ArrayLike = requires(T &&V) {
  { V.data() } -> std::convertible_to<const ValueType *>;
  { V.size() } -> std::same_as<size_t>;
};

template<typename T>
concept DataBuffer = ArrayLike<T, char> || ArrayLike<T, uint8_t>;

//
// append
//
template<typename T>
concept HasReserve = requires(T &&V, size_t S) {
  { V.reserve(S) };
};

template<typename T>
concept HasRangeInsert = requires(T &&V) {
  { V.insert(V.end(), V.begin(), V.end()) };
};

template<std::ranges::sized_range FromType, std::ranges::sized_range ToType>
void append(FromType &&From, ToType &To) {
  // range-based insert is tremendously faster than any other method
  if constexpr (HasRangeInsert<ToType>) {
    To.insert(To.end(), From.begin(), From.end());
    return;
  }

  if constexpr (HasReserve<FromType>)
    To.reserve(To.size() + From.size());

  if constexpr (std::is_lvalue_reference_v<FromType>)
    std::ranges::copy(From, std::inserter(To, To.end()));
  else
    std::ranges::move(From, std::inserter(To, To.end()));
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
  return std::find(std::begin(std::forward<R>(Range)),
                   std::end(std::forward<R>(Range)),
                   Value);
}

/// \note use `llvm::find_if` instead after it's made `constexpr`.
template<typename R, typename CallableType>
constexpr decltype(auto) find_if(R &&Range, CallableType &&Callable) {
  return std::find_if(std::begin(std::forward<R>(Range)),
                      std::end(std::forward<R>(Range)),
                      std::forward<CallableType>(Callable));
}

/// \note use `llvm::find_if_not` instead after it's made `constexpr`.
template<typename R, typename CallableType>
constexpr decltype(auto) find_if_not(R &&Range, CallableType &&Callable) {
  return std::find_if_not(std::begin(std::forward<R>(Range)),
                          std::end(std::forward<R>(Range)),
                          std::forward<CallableType>(Callable));
}

/// \note `std::find_last` is introduced in c++23,
///       replace with the llvm version when it's available.
template<typename R, typename T>
constexpr decltype(auto) find_last(R &&Range, const T &Value) {
  return std::find(std::rbegin(std::forward<R>(Range)),
                   std::rend(std::forward<R>(Range)),
                   Value);
}

/// \note `std::find_last_if` is introduced in c++23,
///       replace with the llvm version when it's available.
template<typename R, typename CallableType>
constexpr decltype(auto) find_last_if(R &&Range, CallableType &&Callable) {
  return std::find_if(std::rbegin(std::forward<R>(Range)),
                      std::rend(std::forward<R>(Range)),
                      std::forward<CallableType>(Callable));
}

/// \note `std::find_last_if_not` is introduced in c++23,
///       replace with the llvm version when it's available.
template<typename R, typename CallableType>
constexpr decltype(auto) find_last_if_not(R &&Range, CallableType &&Callable) {
  return std::find_if_not(std::rbegin(std::forward<R>(Range)),
                          std::rend(std::forward<R>(Range)),
                          std::forward<CallableType>(Callable));
}

/// \note use `llvm::is_contained` instead after it's made `constexpr`.
template<typename R, typename T>
constexpr bool is_contained(R &&Range, const T &Value) {
  return revng::find(std::forward<R>(Range), Value) != std::end(Range);
}

static_assert(is_contained(std::array{ 1, 2, 3 }, 2) == true);
static_assert(is_contained(std::array{ 1, 2, 3 }, 4) == false);

template<typename Range, typename C>
constexpr bool any_of(Range &&R, C &&L) {
  auto Iterator = revng::find_if(std::forward<Range>(R), std::forward<C>(L));
  return Iterator != std::end(R);
}

} // namespace revng

//
// Some other useful small things
//

template<std::size_t ElementCount, std::ranges::forward_range RangeType>
constexpr auto takeAsTuple(RangeType &&R) {
  revng_assert(std::ranges::size(R) >= ElementCount);
  return compile_time::repeat<ElementCount>([&R]<std::size_t I>() -> auto && {
    return *std::next(R.begin(), I);
  });
}

//
// Some views from the STL.
// TODO: remove these after updating the libc++ version.
//
template<typename EnumType>
[[nodiscard]] constexpr std::underlying_type<EnumType>::type
to_underlying(EnumType Value) {
  return static_cast<std::underlying_type<EnumType>::type>(Value);
}

template<typename RangeType>
auto as_rvalue(RangeType &&Range) {
  return llvm::make_range(std::make_move_iterator(Range.begin()),
                          std::make_move_iterator(Range.end()));
}

namespace revng {
namespace detail {

template<typename ToConstruct, typename Begin, typename End>
concept Impl = std::input_or_output_iterator<Begin>
               && std::sentinel_for<End, Begin>
               && std::is_constructible_v<ToConstruct, Begin, End>;
template<typename ToConstruct, typename Range>
concept IteratorConstructible = std::ranges::range<Range>
                                && Impl<ToConstruct,
                                        decltype(std::declval<Range>().begin()),
                                        decltype(std::declval<Range>().end())>;

template<typename Container>
struct ToImpl {
  template<std::ranges::range Range>
  constexpr Container asContainer(Range &&Input) {
    // TODO: extend to support for more than just containers that
    //       provide a double-iterator constructor.
    return Container(Input.begin(), Input.end());
  }
};

} // namespace detail

// NOTE: the implementation here is very crude, but should be good enough until
//       we update to a version of `libc++` with `c++23` support.
template<typename Container>
constexpr detail::ToImpl<Container> to() {
  return detail::ToImpl<Container>();
};

} // namespace revng

template<std::ranges::range Range,
         revng::detail::IteratorConstructible<Range> Container>
constexpr Container operator|(Range &&R, revng::detail::ToImpl<Container> T) {
  return T.asContainer(std::forward<Range>(R));
}
