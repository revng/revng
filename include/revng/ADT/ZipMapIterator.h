#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <iterator>
#include <set>
#include <tuple>
#include <vector>

#include "llvm/ADT/Optional.h"
#include "llvm/ADT/iterator.h"

#include "revng/ADT/KeyedObjectContainer.h"
#include "revng/Support/Assert.h"

//
// has_mapped_type_member
//

namespace detail {

template<typename T>
using eit_mt_t = typename enable_if_type<typename T::mapped_type>::type;

template<typename T>
using eit_vt_t = typename enable_if_type<typename T::value_type>::type;

} // namespace detail

template<class T, class Enable = void>
struct has_mapped_type_member : std::false_type {};

template<class T>
struct has_mapped_type_member<T, detail::eit_mt_t<T>> : std::true_type {};

//
// has_value_type_member
//
template<class T, class Enable = void>
struct has_value_type_member : std::false_type {};
template<class T>
struct has_value_type_member<T, detail::eit_vt_t<T>> : std::true_type {};

//
// has_key_type_member
//
template<class T, class Enable = void>
struct has_key_type_member : std::false_type {};
template<class T>
struct has_key_type_member<T,
                           typename enable_if_type<typename T::key_type>::type>
  : std::true_type {};

//
// is_map_like
//
template<typename T>
constexpr bool is_map_like_v = (has_value_type_member<T>::value
                                and has_key_type_member<T>::value
                                and has_mapped_type_member<T>::value);

template<typename T, typename K = void>
using enable_if_is_map_like_t = std::enable_if_t<is_map_like_v<T>, K>;

//
// is_set_like
//
namespace detail {
template<typename T>
constexpr bool same_key_value_v = std::is_same_v<typename T::key_type,
                                                 typename T::value_type>;
}

template<class T, class Enable = void>
struct same_key_value_types : std::false_type {};

namespace detail {

template<bool B>
using ei_t = std::enable_if_t<B>;

template<typename A, typename B>
constexpr bool same = std::is_same_v<A, B>;

template<typename T>
using ei_skv_t = ei_t<same<typename T::key_type, typename T::value_type>>;

} // namespace detail

template<class T>
struct same_key_value_types<T, detail::ei_skv_t<T>> : std::true_type {};

template<typename T>
constexpr bool is_set_like_v = (has_value_type_member<T>::value
                                and has_key_type_member<T>::value
                                and not has_mapped_type_member<T>::value
                                and same_key_value_types<T>::value);

template<typename T, typename K = void>
using enable_if_is_set_like_t = std::enable_if_t<is_set_like_v<T>, K>;

//
// is_vector_of_pairs
//
template<typename>
struct is_vector_of_pairs : public std::false_type {};

template<typename K, typename V>
struct is_vector_of_pairs<std::vector<std::pair<const K, V>>>
  : public std::true_type {};

template<typename K, typename V>
struct is_vector_of_pairs<const std::vector<std::pair<const K, V>>>
  : public std::true_type {};

namespace {

using namespace std;

static_assert(is_vector_of_pairs<vector<pair<const int, long>>>::value, "");
static_assert(is_vector_of_pairs<const vector<pair<const int, long>>>::value,
              "");

} // namespace

//
// element_pointer_t
//
template<typename T>
using element_pointer_t = decltype(&*std::declval<T>().begin());

template<typename T>
enable_if_is_set_like_t<T, const typename T::key_type &>
keyFromValue(const typename T::value_type &Value) {
  return Value;
}

template<typename T>
enable_if_is_KeyedObjectContainer_t<T, typename T::key_type>
keyFromValue(const typename T::value_type &Value) {
  return KeyedObjectTraits<typename T::value_type>::key(Value);
}

template<typename T>
std::enable_if_t<is_vector_of_pairs<T>::value,
                 const typename T::value_type::first_type &>
keyFromValue(const typename T::value_type &Value) {
  return Value.first;
}

template<typename T>
enable_if_is_map_like_t<T, const typename T::value_type::first_type &>
keyFromValue(const typename T::value_type &Value) {
  return Value.first;
}

template<typename LeftMap, typename RightMap>
struct DefaultComparator {
  template<typename T, typename Q>
  static int compare(const T &LHS, const Q &RHS) {
    auto LHSKey = keyFromValue<LeftMap>(LHS);
    auto RHSKey = keyFromValue<RightMap>(RHS);
    static_assert(std::is_same_v<decltype(LHSKey), decltype(RHSKey)>);
    auto Less = std::less<decltype(LHSKey)>();
    if (Less(LHSKey, RHSKey))
      return -1;
    else if (Less(RHSKey, LHSKey))
      return 1;
    else
      return 0;
  }
};

template<typename LeftMap, typename RightMap>
using zipmap_pair = std::pair<element_pointer_t<LeftMap>,
                              element_pointer_t<RightMap>>;

namespace detail {
template<typename A, typename B>
using fifc = llvm::iterator_facade_base<A, std::forward_iterator_tag, B>;
}

template<typename LeftMap,
         typename RightMap,
         typename Comparator = DefaultComparator<LeftMap, RightMap>>
class ZipMapIterator
  : public detail::fifc<ZipMapIterator<LeftMap, RightMap, Comparator>,
                        const zipmap_pair<LeftMap, RightMap>> {
public:
  template<bool C, typename A, typename B>
  using conditional_t = typename std::conditional<C, A, B>::type;
  using left_inner_iterator = conditional_t<std::is_const_v<LeftMap>,
                                            typename LeftMap::const_iterator,
                                            typename LeftMap::iterator>;
  using right_inner_iterator = conditional_t<std::is_const_v<RightMap>,
                                             typename RightMap::const_iterator,
                                             typename RightMap::iterator>;
  using left_inner_range = llvm::iterator_range<left_inner_iterator>;
  using right_inner_range = llvm::iterator_range<right_inner_iterator>;
  using value_type = zipmap_pair<LeftMap, RightMap>;
  using reference = typename ZipMapIterator::reference;

private:
  value_type Current;
  left_inner_iterator LeftIt;
  const left_inner_iterator EndLeftIt;
  right_inner_iterator RightIt;
  const right_inner_iterator EndRightIt;

public:
  ZipMapIterator(left_inner_range LeftRange, right_inner_range RightRange) :
    LeftIt(LeftRange.begin()),
    EndLeftIt(LeftRange.end()),
    RightIt(RightRange.begin()),
    EndRightIt(RightRange.end()) {

    next();
  }

  ZipMapIterator(left_inner_iterator LeftIt, right_inner_iterator RightIt) :
    ZipMapIterator(llvm::make_range(LeftIt, LeftIt),
                   llvm::make_range(RightIt, RightIt)) {}

  bool operator==(const ZipMapIterator &Other) const {
    revng_assert(EndLeftIt == Other.EndLeftIt);
    revng_assert(EndRightIt == Other.EndRightIt);
    auto ThisTie = std::tie(LeftIt, RightIt, Current);
    auto OtherTie = std::tie(Other.LeftIt, Other.RightIt, Other.Current);
    return ThisTie == OtherTie;
  }

  ZipMapIterator &operator++() {
    next();
    return *this;
  }

  reference operator*() const { return Current; }

private:
  bool leftIsValid() const { return LeftIt != EndLeftIt; }
  bool rightIsValid() const { return RightIt != EndRightIt; }

  void next() {
    if (leftIsValid() and rightIsValid()) {
      switch (Comparator::compare(*LeftIt, *RightIt)) {
      case 0:
        Current = decltype(Current)(&*LeftIt, &*RightIt);
        LeftIt++;
        RightIt++;
        break;

      case -1:
        Current = std::make_pair(&*LeftIt, nullptr);
        LeftIt++;
        break;

      case 1:
        Current = std::make_pair(nullptr, &*RightIt);
        RightIt++;
        break;

      default:
        revng_abort();
      }
    } else if (leftIsValid()) {
      Current = std::make_pair(&*LeftIt, nullptr);
      LeftIt++;
    } else if (rightIsValid()) {
      Current = std::make_pair(nullptr, &*RightIt);
      RightIt++;
    } else {
      Current = std::make_pair(nullptr, nullptr);
    }
  }
};

template<typename LeftMap,
         typename RightMap,
         typename Comparator = DefaultComparator<LeftMap, RightMap>>
inline ZipMapIterator<LeftMap, RightMap, Comparator>
zipmap_begin(LeftMap &Left, RightMap &Right) {
  return ZipMapIterator<LeftMap,
                        RightMap,
                        Comparator>(llvm::make_range(Left.begin(), Left.end()),
                                    llvm::make_range(Right.begin(),
                                                     Right.end()));
}

template<typename LeftMap,
         typename RightMap,
         typename Comparator = DefaultComparator<LeftMap, RightMap>>
inline ZipMapIterator<LeftMap, RightMap, Comparator>
zipmap_end(LeftMap &Left, RightMap &Right) {
  return ZipMapIterator<LeftMap,
                        RightMap,
                        Comparator>(llvm::make_range(Left.end(), Left.end()),
                                    llvm::make_range(Right.end(), Right.end()));
}

template<typename LeftMap,
         typename RightMap,
         typename Comparator = DefaultComparator<LeftMap, RightMap>>
inline llvm::iterator_range<ZipMapIterator<LeftMap, RightMap, Comparator>>
zipmap_range(LeftMap &Left, RightMap &Right) {
  return llvm::make_range(zipmap_begin<LeftMap, RightMap, Comparator>(Left,
                                                                      Right),
                          zipmap_end<LeftMap, RightMap, Comparator>(Left,
                                                                    Right));
}
