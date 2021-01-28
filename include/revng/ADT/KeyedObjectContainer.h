#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <set>

#include "llvm/Support/YAMLTraits.h"

#include "revng/ADT/KeyedObjectTraits.h"
#include "revng/ADT/STLExtras.h"
#include "revng/Support/Assert.h"

template<typename T>
using KOTKey = decltype(KeyedObjectTraits<T>::key(std::declval<T>()));

template<typename T>
using KOTCompare = std::less<const KOTKey<T>>;

template<typename T, class Compare = KOTCompare<T>>
class MutableSet;

template<typename T, class Compare = KOTCompare<T>>
class SortedVector;

//
// is_KeyedObjectContainer
//
namespace detail {

template<typename T>
using no_cv_t = std::remove_cv_t<T>;

template<typename T, template<typename...> class Ref>
constexpr bool is_no_cv_specialization_v = is_specialization_v<no_cv_t<T>, Ref>;

template<typename T>
constexpr bool is_mutableset_v = is_no_cv_specialization_v<T, MutableSet>;

template<typename T>
constexpr bool is_sortedvector_v = is_no_cv_specialization_v<T, SortedVector>;

template<typename T>
constexpr bool is_KOC_v = is_mutableset_v<T> or is_sortedvector_v<T>;

template<typename T, typename K = void>
using enable_if_is_KOC_t = std::enable_if_t<is_KOC_v<T>, K>;

} // namespace detail

template<typename T>
constexpr bool is_KeyedObjectContainer_v = detail::is_KOC_v<T>;

template<typename T, typename K = void>
using enable_if_is_KeyedObjectContainer_t = detail::enable_if_is_KOC_t<T, K>;

static_assert(is_KeyedObjectContainer_v<MutableSet<int>>);
static_assert(is_KeyedObjectContainer_v<SortedVector<int>>);

template<typename T>
struct llvm::yaml::SequenceTraits<T, enable_if_is_KeyedObjectContainer_t<T>> {
  static size_t size(IO &io, T &Seq) { return Seq.size(); }

  class Inserter {
  private:
    using value_type = typename T::value_type;
    using KOT = KeyedObjectTraits<value_type>;
    using key_type = decltype(KOT::key(std::declval<value_type>()));

  private:
    T &Seq;
    decltype(Seq.begin()) It;
    bool IsOutputting;
    std::optional<typename T::BatchInserter> BatchInserter;
    value_type Instance;
    unsigned Index = 0;

  public:
    Inserter(IO &io, T &Seq) :
      Seq(Seq),
      It(Seq.begin()),
      IsOutputting(io.outputting()),
      Instance(KOT::fromKey(key_type())) {

      if constexpr (std::is_const_v<T>) {
        revng_assert(IsOutputting);
      } else {
        if (not IsOutputting)
          BatchInserter.emplace(std::move(Seq.batch_insert()));
      }
    }

    decltype(*It) &preflightElement(unsigned I) {
      revng_assert(Index == I);
      ++Index;

      if (IsOutputting)
        return *(It++);
      else
        return Instance;
    }

    void postflightElement(unsigned) {
      if (not IsOutputting) {
        BatchInserter->insert(Instance);
        Instance = KOT::fromKey(key_type());
      }
    };
  };
};

//
// has_tuple_size
//
template<typename, typename = void>
struct has_tuple_size : std::false_type {};

template<typename T>
struct has_tuple_size<T, std::void_t<decltype(std::tuple_size<T>{})>>
  : std::true_type {};

template<typename T>
constexpr bool has_tuple_size_v = has_tuple_size<T>::value;

static_assert(has_tuple_size_v<std::tuple<>>);
static_assert(!has_tuple_size_v<std::vector<int>>);
static_assert(!has_tuple_size_v<int>);

template<size_t I, typename T, typename K = void>
using enable_if_tuple_end_t = std::enable_if_t<I == std::tuple_size_v<T>, K>;

template<size_t I, typename T, typename K = void>
using enable_if_not_tuple_end_t = std::enable_if_t<I != std::tuple_size_v<T>,
                                                   K>;

template<typename T, typename R = void>
using enable_if_has_tuple_size_t = std::enable_if_t<has_tuple_size_v<T>, R>;

//
// is_iterable
//
namespace tupletree::detail {

using std::begin;
using std::end;

// Require the following:
// * begin/end
// * operator!=
// * operator++
// * operator*
template<typename T>
decltype(begin(std::declval<T &>()) != end(std::declval<T &>()),
         ++std::declval<decltype(begin(std::declval<T &>())) &>(),
         void(*begin(std::declval<T &>())),
         std::true_type{})
is_iterable_impl(int);

template<typename T>
std::false_type is_iterable_impl(...);

} // namespace tupletree::detail

template<typename T>
using is_iterable = decltype(tupletree::detail::is_iterable_impl<T>(0));

template<typename T>
constexpr bool is_iterable_v = is_iterable<T>::value;

static_assert(is_iterable_v<std::vector<int>>);
static_assert(!is_iterable_v<int>);

//
// is_string_like
//
namespace tupletree::detail {
template<typename T>
typename std::is_same<decltype(std::declval<T>().c_str()), const char *>::type
has_c_str(int);

template<typename>
std::false_type has_c_str(...);

} // namespace tupletree::detail

template<typename T>
using has_c_str = typename decltype(tupletree::detail::has_c_str<T>(0))::type;

template<typename T>
constexpr bool has_c_str_v = has_c_str<T>::value;

static_assert(has_c_str_v<llvm::SmallString<4>>);
static_assert(!has_c_str_v<int>);

template<typename T>
constexpr bool
  is_string_like_v = (std::is_convertible_v<std::string, T> or has_c_str_v<T>);

static_assert(is_string_like_v<llvm::SmallString<4>>);
static_assert(is_string_like_v<std::string>);
static_assert(not is_string_like_v<llvm::ArrayRef<int>>);
static_assert(is_string_like_v<llvm::StringRef>);

//
// is_container
//

template<typename T>
constexpr bool is_container_v = is_iterable_v<T> and not is_string_like_v<T>;

namespace detail {
template<typename T>
constexpr bool is_cot_v = is_container_v<T> or has_tuple_size_v<T>;
} // namespace detail

template<typename T>
constexpr bool is_container_or_tuple_v = detail::is_cot_v<T>;

namespace detail {
template<typename T, typename K = void>
using ei_not_cot_t = std::enable_if_t<!is_container_or_tuple_v<T>, K>;
} // namespace detail

template<typename T, typename K = void>
using enable_if_is_not_container_or_tuple_t = detail::ei_not_cot_t<T, K>;

static_assert(is_container_v<std::vector<int>>);
static_assert(is_container_v<std::set<int>>);
static_assert(is_container_v<std::map<int, int>>);
static_assert(is_container_v<llvm::SmallVector<int, 4>>);
static_assert(!is_container_v<std::string>);
static_assert(!is_container_v<llvm::SmallString<4>>);
static_assert(!is_container_v<llvm::StringRef>);

template<typename T, typename K = void>
using enable_if_is_container_t = std::enable_if_t<is_container_v<T>, K>;

template<typename T, typename K = void>
using enable_if_is_not_container_t = std::enable_if_t<!is_container_v<T>, K>;

//
// is_sorted_container
//
// TODO: this is not very nice
namespace detail {

template<typename T>
constexpr bool is_set_v = is_specialization_v<T, std::set>;

template<typename T>
constexpr bool is_sc_v = is_set_v<T> or is_KeyedObjectContainer_v<T>;

} // namespace detail

template<typename T>
constexpr bool is_sorted_container_v = detail::is_sc_v<T>;

namespace detail {
template<typename T>
constexpr bool is_uc_v = is_container_v<T> and not is_sorted_container_v<T>;
}

template<typename T>
constexpr bool is_unsorted_container_v = detail::is_uc_v<T>;

namespace detail {

template<typename T, typename K = void>
using ei_isc_t = std::enable_if_t<is_sorted_container_v<T>, K>;

template<typename T, typename K = void>
using ei_iuc_t = std::enable_if_t<is_unsorted_container_v<T>, K>;

} // namespace detail

template<typename T, typename K = void>
using enable_if_is_sorted_container_t = detail::ei_isc_t<T, K>;

template<typename T, typename K = void>
using enable_if_is_unsorted_container_t = detail::ei_iuc_t<T, K>;
