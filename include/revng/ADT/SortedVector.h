#pragma once

//
// This file is distributed under the MIT License. See LICENSE.mit for details.
//

#include "llvm/ADT/STLExtras.h"

#include "revng/ADT/KeyedObjectContainer.h"
#include "revng/Support/Assert.h"
#include "revng/Support/Debug.h"

template<class ForwardIt, class BinaryPredicate>
ForwardIt
unique_last(ForwardIt First, ForwardIt Last, BinaryPredicate Predicate) {
  if (First == Last)
    return Last;

  ForwardIt Result = First;
  while (++First != Last) {
    if (not Predicate(*Result, *First) and ++Result != First) {
      *Result = std::move(*First);
    } else {
      *Result = std::move(*First);
    }
  }

  return ++Result;
}

template<KeyedObjectContainerCompatible T,
         class Compare = DefaultKeyObjectComparator<T>>
class SortedVector {
public:
  using KOT = KeyedObjectTraits<T>;
  using key_type = const decltype(KOT::key(std::declval<T>()));

private:
  using vector_type = std::vector<T>;

public:
  using size_type = typename vector_type::size_type;
  using value_type = typename vector_type::value_type;
  using difference_type = typename vector_type::difference_type;
  using allocator_type = typename vector_type::allocator_type;
  using reference = typename vector_type::reference;
  using const_reference = typename vector_type::const_reference;
  using pointer = typename vector_type::pointer;
  using const_pointer = typename vector_type::const_pointer;

public:
  using iterator = typename vector_type::iterator;
  using const_iterator = typename vector_type::const_iterator;
  using reverse_iterator = typename vector_type::reverse_iterator;
  using const_reverse_iterator = typename vector_type::const_reverse_iterator;

public:
  static constexpr bool KeyedObjectContainerTag = true;

private:
  vector_type TheVector;
  bool BatchInsertInProgress = false;

public:
  SortedVector() {}

  SortedVector(std::initializer_list<T> List) {
    auto Inserter = batch_insert_or_assign();
    for (const T &Element : List) {
      Inserter.insert_or_assign(Element);
    }
  }

  template<std::input_iterator FromIt, std::sentinel_for<FromIt> ToIt>
  SortedVector(FromIt From, ToIt To) : TheVector(From, To) {
    sort<true>();
  }

public:
  void swap(SortedVector &Other) {
    revng_assert(not BatchInsertInProgress);
    TheVector.swap(Other.TheVector);
  }

  bool operator==(const SortedVector &) const = default;

public:
  T &at(const key_type &Key) {
    revng_assert(not BatchInsertInProgress);
    auto It = find(Key);
    revng_assert(It != end());
    return *It;
  }

  const T &at(const key_type &Key) const {
    revng_assert(not BatchInsertInProgress);
    auto It = find(Key);
    revng_assert(It != end());
    return *It;
  }

  T &operator[](const key_type &Key) {
    revng_assert(not BatchInsertInProgress);
    return *insert(KOT::fromKey(Key)).first;
  }

  T &operator[](key_type &&Key) {
    revng_assert(not BatchInsertInProgress);
    return *insert(KOT::fromKey(Key)).first;
  }

  iterator begin() {
    revng_assert(not BatchInsertInProgress);
    return TheVector.begin();
  }

  iterator end() {
    revng_assert(not BatchInsertInProgress);
    return TheVector.end();
  }

  const_iterator begin() const {
    revng_assert(not BatchInsertInProgress);
    return TheVector.begin();
  }

  const_iterator end() const {
    revng_assert(not BatchInsertInProgress);
    return TheVector.end();
  }

  const_iterator cbegin() const {
    revng_assert(not BatchInsertInProgress);
    return TheVector.cbegin();
  }

  const_iterator cend() const {
    revng_assert(not BatchInsertInProgress);
    return TheVector.cend();
  }

  reverse_iterator rbegin() {
    revng_assert(not BatchInsertInProgress);
    return TheVector.rbegin();
  }

  reverse_iterator rend() {
    revng_assert(not BatchInsertInProgress);
    return TheVector.rend();
  }

  const_reverse_iterator rbegin() const {
    revng_assert(not BatchInsertInProgress);
    return TheVector.rbegin();
  }

  const_reverse_iterator rend() const {
    revng_assert(not BatchInsertInProgress);
    return TheVector.rend();
  }

  const_reverse_iterator crbegin() const {
    revng_assert(not BatchInsertInProgress);
    return TheVector.crbegin();
  }

  const_reverse_iterator crend() const {
    revng_assert(not BatchInsertInProgress);
    return TheVector.crend();
  }

  bool empty() const {
    revng_assert(not BatchInsertInProgress);
    return TheVector.empty();
  }

  size_type size() const {
    revng_assert(not BatchInsertInProgress);
    return TheVector.size();
  }

  size_type max_size() const {
    revng_assert(not BatchInsertInProgress);
    return TheVector.max_size();
  }

  void clear() {
    revng_assert(not BatchInsertInProgress);
    TheVector.clear();
  }

  void reserve(size_type NewSize) {
    revng_assert(not BatchInsertInProgress);
    TheVector.reserve(NewSize);
  }

  size_type capacity() const {
    revng_assert(not BatchInsertInProgress);
    return TheVector.capacity();
  }

  std::pair<iterator, bool> insert(const T &Value) { return emplace(Value); }

  template<typename... Types>
  std::pair<iterator, bool> emplace(Types &&...Values) {
    revng_assert(not BatchInsertInProgress);

    T Value{ std::forward<Types>(Values)... };
    auto Key = KeyedObjectTraits<T>::key(Value);
    auto It = lower_bound(Key);
    if (It == end()) {
      TheVector.emplace_back(std::move(Value));
      return { --end(), true };
    } else if (keysEqual(KeyedObjectTraits<T>::key(*It), Key)) {
      return { It, false };
    } else {
      return { TheVector.emplace(It, std::move(Value)), true };
    }
  }

  std::pair<iterator, bool> insert_or_assign(const T &Value) {
    return emplace_or_assign(Value);
  }

  template<typename... Types> // NOLINTNEXTLINE
  std::pair<iterator, bool> emplace_or_assign(Types &&...Values) {
    revng_assert(not BatchInsertInProgress);

    T Value{ std::forward<Types>(Values)... };
    auto Key = KeyedObjectTraits<T>::key(Value);
    auto It = lower_bound(Key);
    if (It == end()) {
      TheVector.emplace_back(std::move(Value));
      return { --end(), true };
    } else if (keysEqual(KeyedObjectTraits<T>::key(*It), Key)) {
      *It = std::move(Value);
      return { It, false };
    } else {
      return { TheVector.emplace(It, std::move(Value)), true };
    }
  }

  iterator erase(iterator Pos) {
    revng_assert(not BatchInsertInProgress);
    return TheVector.erase(Pos);
  }

  iterator erase(const_iterator First, const_iterator Last) {
    revng_assert(not BatchInsertInProgress);
    return TheVector.erase(First, Last);
  }

  size_type erase(const key_type &Key) {
    revng_assert(not BatchInsertInProgress);
    auto It = find(Key);
    if (It == end()) {
      return 0;
    } else {
      erase(It);
      return 1;
    }
  }

  size_type count(const key_type &Key) const {
    revng_assert(not BatchInsertInProgress);
    return find(Key) != end() ? 1 : 0;
  }

  iterator find(const key_type &Key) {
    revng_assert(not BatchInsertInProgress);
    auto It = lower_bound(Key);
    auto End = end();
    if (!(It == End) and Compare()(Key, KeyedObjectTraits<T>::key(*It))) {
      return End;
    } else {
      return It;
    }
  }

  const_iterator find(const key_type &Key) const {
    revng_assert(not BatchInsertInProgress);
    auto It = lower_bound(Key);
    auto End = end();
    if (!(It == End) and Compare()(Key, KeyedObjectTraits<T>::key(*It))) {
      return End;
    } else {
      return It;
    }
  }

  bool contains(const key_type &Key) const { return find(Key) != end(); }

  iterator lower_bound(const key_type &Key) {
    revng_assert(not BatchInsertInProgress);
    return std::lower_bound(begin(), end(), KOT::fromKey(Key), compareElements);
  }

  const_iterator lower_bound(const key_type &Key) const {
    revng_assert(not BatchInsertInProgress);
    return std::lower_bound(begin(), end(), KOT::fromKey(Key), compareElements);
  }

  iterator upper_bound(const key_type &Key) {
    revng_assert(not BatchInsertInProgress);
    return std::upper_bound(begin(), end(), KOT::fromKey(Key), compareElements);
  }

  const_iterator upper_bound(const key_type &Key) const {
    revng_assert(not BatchInsertInProgress);
    return std::upper_bound(begin(), end(), KOT::fromKey(Key), compareElements);
  }

public:
  template<bool EnsureUnique = false>
  class BatchInserterBase {
  private:
    SortedVector *SV;

  public:
    BatchInserterBase(SortedVector &SV) : SV(&SV) {
      revng_assert(not SV.BatchInsertInProgress);
      SV.BatchInsertInProgress = true;
    }

    BatchInserterBase(const BatchInserterBase &) = delete;
    BatchInserterBase &operator=(const BatchInserterBase &) = delete;

    BatchInserterBase(BatchInserterBase &&Other) {
      SV = Other.SV;
      Other.SV = nullptr;
    }

    BatchInserterBase &operator=(BatchInserterBase &&Other) {
      SV = Other.SV;
      Other.SV = nullptr;
    }

    ~BatchInserterBase() { commit(); }

    void commit() {
      if (SV != nullptr && SV->BatchInsertInProgress) {
        SV->BatchInsertInProgress = false;
        SV->sort<EnsureUnique>();
      }
    }

  protected:
    template<typename... Types>
    T &emplaceImpl(Types &&...Values) {
      revng_assert(SV->BatchInsertInProgress);
      SV->TheVector.emplace_back(std::forward<Types>(Values)...);
      return SV->TheVector.back();
    }
  };

  class BatchInserter : public BatchInserterBase<true> {
  public:
    BatchInserter(SortedVector &SV) : BatchInserterBase<true>(SV) {}

  public:
    template<typename... Types>
    T &emplace(Types &&...Values) {
      return this->emplaceImpl(std::forward<Types>(Values)...);
    }
    T &insert(const T &Value) { return emplace(Value); }
  };

  BatchInserter batch_insert() {
    revng_assert(not BatchInsertInProgress);
    return BatchInserter(*this);
  }

  class BatchInsertOrAssigner : public BatchInserterBase<false> {
  public:
    BatchInsertOrAssigner(SortedVector &SV) : BatchInserterBase<false>(SV) {}

  public:
    template<typename... Types> // NOLINTNEXTLINE
    T &emplace_or_assign(Types &&...Values) {
      return this->emplaceImpl(std::forward<Types>(Values)...);
    }
    T &insert_or_assign(const T &Value) { return emplace_or_assign(Value); }
  };

  BatchInsertOrAssigner batch_insert_or_assign() {
    revng_assert(not BatchInsertInProgress);
    return BatchInsertOrAssigner(*this);
  }

  /// \note This function should always return true
  bool isSorted() const debug_function {
    auto It = begin();
    if (It == end())
      return true;

    while (true) {
      const auto &Left = *It;
      ++It;
      if (It == end())
        return true;
      const auto &Right = *It;

      if (not compareElements(Left, Right))
        return false;
    }
  }

private:
  static bool compareElements(const T &LHS, const T &RHS) {
    return compareKeys(KeyedObjectTraits<T>::key(LHS),
                       KeyedObjectTraits<T>::key(RHS));
  }

  static bool compareKeys(const key_type &LHS, const key_type &RHS) {
    return Compare()(LHS, RHS);
  }

  static bool elementsEqual(const T &LHS, const T &RHS) {
    return keysEqual(KeyedObjectTraits<T>::key(LHS),
                     KeyedObjectTraits<T>::key(RHS));
  }

  static bool keysEqual(const key_type &LHS, const key_type &RHS) {
    return not compareKeys(LHS, RHS) and not compareKeys(RHS, LHS);
  }

  template<bool EnsureUnique>
  void sort() {
    if constexpr (EnsureUnique) {
      std::sort(begin(), end(), compareElements);
      revng_check(std::adjacent_find(begin(), end(), elementsEqual) == end(),
                  "Multiples of the same element in a `SortedVector`.");
    } else {
      std::stable_sort(begin(), end(), compareElements);
      auto NewEnd = unique_last(begin(), end(), elementsEqual);
      TheVector.erase(NewEnd, end());
    }
  }
};

static_assert(KeyedObjectContainer<SortedVector<int>>);
