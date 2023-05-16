#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <map>

#include "llvm/ADT/STLExtras.h"

#include "revng/ADT/KeyedObjectContainer.h"

template<KeyedObjectContainerCompatible T,
         class Compare = DefaultKeyObjectComparator<T>>
class MutableSet {
private:
  using KOT = KeyedObjectTraits<T>;
  using non_const_key_type = decltype(KOT::key(std::declval<T>()));

public:
  using key_type = const non_const_key_type;

private:
  using map_type = std::map<non_const_key_type, T, Compare>;

public:
  using size_type = typename map_type::size_type;
  using value_type = T;
  using difference_type = typename map_type::difference_type;
  using allocator_type = typename map_type::allocator_type;
  using reference = T &;
  using const_reference = const T &;
  using pointer = T *;
  using const_pointer = const T *;

private:
  using inner_mapped_type = typename map_type::mapped_type;
  using inner_iterator = typename map_type::iterator;
  using const_inner_iterator = typename map_type::const_iterator;
  using inner_reverse_iterator = typename map_type::reverse_iterator;
  using ciri = typename map_type::const_reverse_iterator;
  using const_inner_reverse_iterator = ciri;

private:
  template<typename A, typename B>
  using mapped_iterator = llvm::mapped_iterator<A, B>;

private:
  using pair = std::pair<key_type, inner_mapped_type>;
  static inner_mapped_type &getSecond(pair &Pair) { return Pair.second; }

  static const inner_mapped_type &getConstSecond(const pair &Pair) {
    return Pair.second;
  }

public:
  using iterator = mapped_iterator<inner_iterator, decltype(getSecond) *>;
  using const_iterator = mapped_iterator<const_inner_iterator,
                                         decltype(getConstSecond) *>;
  using reverse_iterator = mapped_iterator<inner_reverse_iterator,
                                           decltype(getSecond) *>;
  using const_reverse_iterator = mapped_iterator<const_inner_reverse_iterator,
                                                 decltype(getConstSecond) *>;

public:
  static constexpr bool KeyedObjectContainerTag = true;

private:
  map_type TheMap;

public:
  MutableSet() {}

  MutableSet(std::initializer_list<T> List) {
    auto Inserter = batch_insert_or_assign();
    for (const T &Element : List) {
      Inserter.insert_or_assign(Element);
    }
  }

public:
  void swap(MutableSet &Other) { TheMap.swap(Other.TheMap); }
  bool operator==(const MutableSet &Other) const = default;

public:
  T &at(const key_type &Key) { return TheMap.at(Key); }
  const T &at(const key_type &Key) const { return TheMap.at(Key); }

  T &operator[](const key_type &Key) {
    return TheMap.insert({ Key, KOT::fromKey(Key) }).first->second;
  }

  T &operator[](key_type &&Key) {
    return TheMap.insert({ Key, KOT::fromKey(Key) }).first->second;
  }

  iterator begin() { return wrapIterator(TheMap.begin()); }
  iterator end() { return wrapIterator(TheMap.end()); }
  const_iterator begin() const { return wrapIterator(TheMap.begin()); }
  const_iterator end() const { return wrapIterator(TheMap.end()); }
  const_iterator cbegin() const { return wrapIterator(TheMap.cbegin()); }
  const_iterator cend() const { return wrapIterator(TheMap.cend()); }
  reverse_iterator rbegin() { return wrapIterator(TheMap.rbegin()); }
  reverse_iterator rend() { return wrapIterator(TheMap.rend()); }
  const_reverse_iterator rbegin() const {
    return wrapIterator(TheMap.rbegin());
  }
  const_reverse_iterator rend() const { return wrapIterator(TheMap.rend()); }
  const_reverse_iterator crbegin() const {
    return wrapIterator(TheMap.crbegin());
  }
  const_reverse_iterator crend() const { return wrapIterator(TheMap.crend()); }

  bool empty() const { return TheMap.empty(); }
  size_type size() const { return TheMap.size(); }
  size_type max_size() const { return TheMap.max_size(); }
  void clear() { TheMap.clear(); }

  std::pair<iterator, bool> insert(const T &Value) {
    auto Result = TheMap.insert({ KOT::key(Value), Value });
    return { wrapIterator(Result.first), Result.second };
  }

  template<typename... Types>
  std::pair<iterator, bool> emplace(Types &&...Values) {
    T NewElement{ Values... };

    auto Result = TheMap.emplace(KOT::key(NewElement), std::move(NewElement));

    return { wrapIterator(Result.first), Result.second };
  }

  template<typename... Types>
  std::pair<iterator, bool> try_emplace(Types &&...Values) {
    T NewElement{ Values... };

    auto Result = TheMap.try_emplace(KOT::key(NewElement),
                                     std::move(NewElement));

    return { wrapIterator(Result.first), Result.second };
  }

  std::pair<iterator, bool> insert_or_assign(const T &Value) {
    auto Result = TheMap.insert_or_assign(KOT::key(Value), Value);
    return { wrapIterator(Result.first), Result.second };
  }

  iterator erase(iterator Pos) {
    return wrapIterator(TheMap.erase(unwrapIterator(Pos)));
  }
  iterator erase(iterator First, iterator Last) {
    auto It = TheMap.erase(unwrapIterator(First), unwrapIterator(Last));
    return wrapIterator(It);
  }

  size_type erase(const key_type &Key) { return TheMap.erase(Key); }

  size_type count(const key_type &Key) const { return TheMap.count(Key); }

  iterator find(const key_type &Key) { return wrapIterator(TheMap.find(Key)); }

  const_iterator find(const key_type &Key) const {
    return wrapIterator(TheMap.find(Key));
  }

  bool contains(const key_type &Key) const { return find(Key) != end(); }

  iterator lower_bound(const key_type &Key) {
    return wrapIterator(TheMap.lower_bound(Key));
  }

  const_iterator lower_bound(const key_type &Key) const {
    return wrapIterator(TheMap.lower_bound(Key));
  }

  iterator upper_bound(const key_type &Key) {
    return wrapIterator(TheMap.upper_bound(Key));
  }

  const_iterator upper_bound(const key_type &Key) const {
    return wrapIterator(TheMap.upper_bound(Key));
  }

public:
  class BatchInserter {
  private:
    MutableSet &MS;

  public:
    BatchInserter(MutableSet &MS) : MS(MS) {}
    T &insert(const T &Value) { return *MS.insert(Value).first; }
  };

  BatchInserter batch_insert() { return BatchInserter(*this); }

  class BatchInsertOrAssigner {
  private:
    MutableSet &MS;

  public:
    BatchInsertOrAssigner(MutableSet &MS) : MS(MS) {}
    T &insert_or_assign(const T &Value) {
      return *MS.insert_or_assign(Value).first;
    }
  };

  BatchInsertOrAssigner batch_insert_or_assign() {
    return BatchInsertOrAssigner(*this);
  }

private:
  static inner_iterator unwrapIterator(iterator It) { return It.getCurrent(); }

  static const_inner_iterator unwrapIterator(const_iterator It) {
    return It.getCurrent();
  }

  static iterator wrapIterator(inner_iterator It) {
    return iterator(It, getSecond);
  }

  static const_iterator wrapIterator(const_inner_iterator It) {
    return const_iterator(It, getConstSecond);
  }

  static reverse_iterator wrapIterator(inner_reverse_iterator It) {
    return reverse_iterator(It, getSecond);
  }

  static const_reverse_iterator wrapIterator(const_inner_reverse_iterator It) {
    return const_reverse_iterator(It, getConstSecond);
  }
};

static_assert(KeyedObjectContainer<MutableSet<int>>);
