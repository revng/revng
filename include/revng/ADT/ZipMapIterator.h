#ifndef ZIPMAPITERATOR_H
#define ZIPMAPITERATOR_H

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// Standard includes
#include <iterator>
#include <set>
#include <tuple>
#include <vector>

// LLVM includes
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/iterator.h"

// Local libraries includes
#include "revng/Support/Assert.h"

// For std::map-compatible containers
template<typename T, typename = void>
struct KeyContainer {
  using key_type = typename T::key_type;
  using pointer = std::conditional_t<std::is_const<T>::value,
                                     typename T::const_pointer,
                                     typename T::pointer>;
  using value_type = std::conditional_t<std::is_const<T>::value,
                                        const typename T::value_type,
                                        typename T::value_type>;

  static const typename T::key_type &getKey(value_type &Value) {
    return Value.first;
  }

  static void insert(T &Container, typename T::key_type Key) {
    Container.insert({ Key, typename T::mapped_type() });
  }

  static pointer find(T &Container, typename T::key_type &Key) {
    return &*Container.find(Key);
  }

  static void sort(T &) {}
};

template<typename>
struct isSet : public std::false_type {};

template<typename T>
struct isSet<std::set<T>> : public std::true_type {};

template<typename T>
struct isSet<const std::set<T>> : public std::true_type {};

static_assert(isSet<std::set<int>>::value, "");
static_assert(isSet<const std::set<int>>::value, "");

template<typename T>
struct KeyContainer<T, typename std::enable_if_t<isSet<T>::value>> {
  using key_type = typename T::key_type;
  using pointer = const key_type *;

  static const key_type &getKey(const key_type &Value) { return Value; }

  static void insert(T &Container, key_type Key) { Container.insert(Key); }

  static pointer find(const T &Container, key_type Key) {
    return &*Container.find(Key);
  }

  static void sort(T &) {}
};

template<typename>
struct isVectorOfPairs : public std::false_type {};

template<typename K, typename V>
struct isVectorOfPairs<std::vector<std::pair<const K, V>>>
  : public std::true_type {};

template<typename K, typename V>
struct isVectorOfPairs<const std::vector<std::pair<const K, V>>>
  : public std::true_type {};

namespace {

using namespace std;

static_assert(isVectorOfPairs<vector<pair<const int, long>>>::value, "");
static_assert(isVectorOfPairs<const vector<pair<const int, long>>>::value, "");

} // namespace

template<typename T>
struct KeyContainer<T, typename std::enable_if_t<isVectorOfPairs<T>::value>> {
  using key_type = typename T::value_type::first_type;
  using value_type = std::conditional_t<std::is_const<T>::value,
                                        const typename T::value_type,
                                        typename T::value_type>;
  using pointer = std::conditional_t<std::is_const<T>::value,
                                     typename T::const_pointer,
                                     typename T::pointer>;
  using mapped_type = typename value_type::second_type;

  static const key_type &getKey(value_type &Value) { return Value.first; }

  static void insert(T &Container, key_type Key) {
    Container.push_back({ Key, mapped_type() });
  }

  static pointer find(T &Container, key_type Key) {
    auto Condition = [Key](const value_type &Value) {
      return Value.first == Key;
    };
    return &*std::find_if(Container.begin(), Container.end(), Condition);
  }

  static void sort(T &Container) {
    static_assert(not std::is_const<T>::value, "");
    using non_const_value_type = std::pair<std::remove_const_t<key_type>,
                                           mapped_type>;
    auto Less = [](const non_const_value_type &This,
                   const non_const_value_type &Other) {
      return This.first < Other.first;
    };
    using vector = std::vector<non_const_value_type>;
    auto &NonConst = *reinterpret_cast<vector *>(&Container);
    std::sort(NonConst.begin(), NonConst.end(), Less);
  }
};

template<typename Map>
using zipmap_pair = std::pair<typename KeyContainer<Map>::pointer,
                              typename KeyContainer<Map>::pointer>;

template<typename Map>
class ZipMapIterator
  : public llvm::iterator_facade_base<ZipMapIterator<Map>,
                                      std::forward_iterator_tag,
                                      const zipmap_pair<Map>> {
public:
  template<bool C, typename A, typename B>
  using conditional_t = typename std::conditional<C, A, B>::type;
  using inner_iterator = conditional_t<std::is_const<Map>::value,
                                       typename Map::const_iterator,
                                       typename Map::iterator>;
  using inner_range = llvm::iterator_range<inner_iterator>;
  using value_type = zipmap_pair<Map>;
  using reference = typename ZipMapIterator::reference;

private:
  value_type Current;
  inner_iterator LeftIt;
  const inner_iterator EndLeftIt;
  inner_iterator RightIt;
  const inner_iterator EndRightIt;

public:
  ZipMapIterator(inner_range LeftRange, inner_range RightRange) :
    LeftIt(LeftRange.begin()),
    EndLeftIt(LeftRange.end()),
    RightIt(RightRange.begin()),
    EndRightIt(RightRange.end()) {

    next();
  }

  ZipMapIterator(inner_iterator LeftIt, inner_iterator RightIt) :
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
  static const typename KeyContainer<Map>::key_type &getKey(inner_iterator It) {
    return KeyContainer<Map>::getKey(*It);
  }

  bool leftIsValid() const { return LeftIt != EndLeftIt; }
  bool rightIsValid() const { return RightIt != EndRightIt; }

  void next() {
    if (leftIsValid() and rightIsValid()) {
      if (getKey(LeftIt) == getKey(RightIt)) {
        Current = std::make_pair(&*LeftIt, &*RightIt);
        LeftIt++;
        RightIt++;
      } else if (getKey(LeftIt) < getKey(RightIt)) {
        Current = std::make_pair(&*LeftIt, nullptr);
        LeftIt++;
      } else {
        Current = std::make_pair(nullptr, &*RightIt);
        RightIt++;
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

template<typename T>
inline ZipMapIterator<T> zipmap_begin(T &Left, T &Right) {
  return ZipMapIterator<T>(llvm::make_range(Left.begin(), Left.end()),
                           llvm::make_range(Right.begin(), Right.end()));
}

template<typename T>
inline ZipMapIterator<T> zipmap_end(T &Left, T &Right) {
  return ZipMapIterator<T>(llvm::make_range(Left.end(), Left.end()),
                           llvm::make_range(Right.end(), Right.end()));
}

template<typename T>
inline llvm::iterator_range<ZipMapIterator<T>> zipmap_range(T &Left, T &Right) {
  return llvm::make_range(zipmap_begin(Left, Right), zipmap_end(Left, Right));
}

#endif // ZIPMAPITERATOR_H
