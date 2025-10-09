#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <stack>
#include <type_traits>
#include <vector>

#include "llvm/ADT/BitVector.h"

#include "revng/ADT/KeyedObjectContainer.h"
#include "revng/ADT/MutableSet.h"
#include "revng/ADT/SortedVector.h"
#include "revng/ADT/UpcastablePointer.h"
#include "revng/Support/AccessTracker.h"
#include "revng/Support/Generator.h"
#include "revng/TupleTree/TupleLikeTraits.h"

namespace revng {
struct TrackingImpl;
} // namespace revng

namespace revng {

namespace detail {
template<typename V>
concept HasName = requires() {
  { TupleLikeTraits<V>::Name } -> std::convertible_to<llvm::StringRef>;
};
}

/// Wrapper around a KeyedObjectContainer tracking accessed elements
///
/// It allows to keep track of which elements have been read from the container,
/// which elements were required from the container but were not found. Should a
/// getter that allows to inspect the entire content of the container be
/// invoked, such as find, begin and end, then the entire container is marked as
/// accessed.
template<KeyedObjectContainer T>
class TrackingContainer {
public:
  using UnderlyingContainer = T;

public:
  friend struct revng::TrackingImpl;

  using key_type = typename T::key_type;
  using TrackingSet = std::set<std::remove_const_t<key_type>>;
  using size_type = typename T::size_type;
  using value_type = typename T::value_type;
  using difference_type = typename T::difference_type;
  using allocator_type = typename T::allocator_type;
  using reference = typename T::reference;
  using const_reference = typename T::const_reference;
  using pointer = typename T::pointer;
  using const_pointer = typename T::const_pointer;
  using iterator = typename T::iterator;
  using const_iterator = typename T::const_iterator;
  using reverse_iterator = typename T::reverse_iterator;
  using const_reverse_iterator = typename T::const_reverse_iterator;

  struct TrackingResult {
    bool Exact;
    TrackingSet InspectedKeys;
  };

public:
  static constexpr bool KeyedObjectContainerTag = true;

private:
  T Content;
  mutable AccessTracker Exact = AccessTracker(false);
  mutable std::vector<TrackingSet> NonExisting;
  mutable std::vector<llvm::BitVector> Existing;

  mutable bool TrackingIsActive = false;

public:
  /// \defgroup Methods related to tracking
  /// @{

  void stopTracking() const { TrackingIsActive = false; }
  void clearTracking() const {
    Exact.clear();
    NonExisting = {};
    NonExisting.push_back(TrackingSet());
    Existing = {};
    Existing.push_back(llvm::BitVector(Content.size()));
    TrackingIsActive = true;
  }

  TrackingResult getTrackingResult() const {
    TrackingResult ToReturn;

    llvm::copy(getExistingRequestedKeys(),
               std::inserter(ToReturn.InspectedKeys,
                             ToReturn.InspectedKeys.begin()));
    llvm::copy(getNonExistingRequestedKeys(),
               std::inserter(ToReturn.InspectedKeys,
                             ToReturn.InspectedKeys.begin()));

    ToReturn.Exact = Exact.isSet();

    return ToReturn;
  }

  void trackingPush() const {
    Existing.push_back(llvm::BitVector(Content.size()));
    NonExisting.push_back(TrackingSet());
    Exact.push();
  }

  void trackingPop() const {
    Existing.pop_back();
    if (Existing.empty())
      Existing.push_back(llvm::BitVector(Content.size()));

    NonExisting.pop_back();
    if (NonExisting.empty())
      NonExisting.push_back(TrackingSet());

    Exact.pop();
  }

  /// @}

public:
  /// \defgroup Non const methods not trigger the access tracking mechanism,
  ///           both for performance reasons and because it does not make
  ///           semantically much sense.
  /// @{
  TrackingContainer() {}

  TrackingContainer(T &&Underlying) : T(std::move(Underlying)) {}

  TrackingContainer(std::initializer_list<T> List) : T(std::move(List)) {}

  TrackingContainer(TrackingContainer &&) = default;
  TrackingContainer(const TrackingContainer &) = default;
  ~TrackingContainer() = default;
  TrackingContainer &operator=(TrackingContainer &&) = default;
  TrackingContainer &operator=(const TrackingContainer &) = default;
  TrackingContainer &operator=(const T &Other) {
    Content = Other;
    return *this;
  }
  TrackingContainer &operator=(T &&Other) {
    Content = std::move(Other);
    return *this;
  }

  void swap(TrackingContainer &Other) { Content.swap(Other.Content); }

  value_type &at(const key_type &Key) { return Content.at(Key); }

  value_type &operator[](key_type &&Key) { return Content[Key]; }

  iterator begin() { return Content.begin(); }

  iterator end() { return Content.end(); }

  reverse_iterator rbegin() { return Content.rbegin(); }

  reverse_iterator rend() { return Content.rend(); }

  void clear() { Content.clear(); }

  void reserve(size_type NewSize) { Content.reserve(NewSize); }

  std::pair<iterator, bool> insert(const value_type &Value) {
    return Content.insert(Value);
  }

  template<typename... Types>
  std::pair<iterator, bool> emplace(Types &&...Values) {
    return Content.emplace(std::forward<Types>(Values)...);
  }

  std::pair<iterator, bool> insert_or_assign(const value_type &Value) {
    return Content.insert_or_assign(Value);
  }

  iterator erase(iterator Pos) { return Content.erase(Pos); }

  iterator erase(iterator First, iterator Last) {
    return Content.erase(First, Last);
  }

  size_type erase(const key_type &Key) { return Content.erase(Key); }

  template<typename CallableType>
  size_type erase_if(CallableType &&Callable) {
    return Content.erase_if(std::forward<CallableType>(Callable));
  }

  iterator find(const key_type &Key) { return Content.find(Key); }

  iterator lower_bound(const key_type &Key) { return Content.lower_bound(Key); }

  iterator upper_bound(const key_type &Key) { return Content.upper_bound(Key); }
  /// @}

public:
  /// \defgroup const methods trigger the access tracking mechanism
  /// @{

  bool operator==(const TrackingContainer &Other) const {
#ifdef TUPLE_TREE_GENERATOR_EMIT_TRACKING_DEBUG
    onFieldAccess("operator==", name());
#endif
    Exact.access();
    Other.Exact.access();
    return Content == Other.Content;
  }

  bool operator==(const T &Other) const {
#ifdef TUPLE_TREE_GENERATOR_EMIT_TRACKING_DEBUG
    onFieldAccess("operator==", name());
#endif
    Exact.access();
    return Content == Other;
  }

  auto operator<=>(const TrackingContainer &Other) const {
#ifdef TUPLE_TREE_GENERATOR_EMIT_TRACKING_DEBUG
    onFieldAccess("operator<=>", name());
#endif
    Exact.access();
    Other.Exact.access();
    return Content <=> Other.Content;
  }

  auto operator<=>(const T &Other) const {
#ifdef TUPLE_TREE_GENERATOR_EMIT_TRACKING_DEBUG
    onFieldAccess("operator<=>", name());
#endif
    Exact.access();
    return Content <=> Other;
  }

  const value_type &at(const key_type &Key) const {
#ifdef TUPLE_TREE_GENERATOR_EMIT_TRACKING_DEBUG
    onFieldAccess("at", name());
#endif
    markExistingKey(Key);
    return Content.at(Key);
  }

  value_type &operator[](const key_type &Key) { return Content[Key]; }

  const_iterator begin() const {
#ifdef TUPLE_TREE_GENERATOR_EMIT_TRACKING_DEBUG
    onFieldAccess("begin", name());
#endif
    Exact.access();
    return Content.begin();
  }

  const_iterator end() const {
#ifdef TUPLE_TREE_GENERATOR_EMIT_TRACKING_DEBUG
    onFieldAccess("end", name());
#endif
    Exact.access();
    return Content.end();
  }

  const_iterator cbegin() const {
#ifdef TUPLE_TREE_GENERATOR_EMIT_TRACKING_DEBUG
    onFieldAccess("cbegin", name());
#endif
    Exact.access();
    return Content.cbegin();
  }

  const_iterator cend() const {
#ifdef TUPLE_TREE_GENERATOR_EMIT_TRACKING_DEBUG
    onFieldAccess("cend", name());
#endif
    Exact.access();
    return Content.cend();
  }
  const_reverse_iterator rbegin() const {
#ifdef TUPLE_TREE_GENERATOR_EMIT_TRACKING_DEBUG
    onFieldAccess("rbegin", name());
#endif
    Exact.access();
    return Content.rbegin();
  }

  const_reverse_iterator rend() const {
#ifdef TUPLE_TREE_GENERATOR_EMIT_TRACKING_DEBUG
    onFieldAccess("rend", name());
#endif
    Exact.access();
    return Content.rend();
  }

  const_reverse_iterator crbegin() const {
#ifdef TUPLE_TREE_GENERATOR_EMIT_TRACKING_DEBUG
    onFieldAccess("crbegin", name());
#endif
    Exact.access();
    return Content.crbegin();
  }

  const_reverse_iterator crend() const {
#ifdef TUPLE_TREE_GENERATOR_EMIT_TRACKING_DEBUG
    onFieldAccess("crend", name());
#endif
    Exact.access();
    return Content.crend();
  }

  bool empty() const {
#ifdef TUPLE_TREE_GENERATOR_EMIT_TRACKING_DEBUG
    onFieldAccess("empty", name());
#endif
    Exact.access();
    return Content.empty();
  }

  size_type size() const {
#ifdef TUPLE_TREE_GENERATOR_EMIT_TRACKING_DEBUG
    onFieldAccess("size", name());
#endif
    Exact.access();
    return Content.size();
  }

  size_type max_size() const { return Content.max_size(); }

  size_type capacity() const { return Content.capacity(); }

  size_type count(const key_type &Key) const {
#ifdef TUPLE_TREE_GENERATOR_EMIT_TRACKING_DEBUG
    onFieldAccess("count", name());
#endif
    auto Quantity = Content.count(Key);

    if (not TrackingIsActive)
      return Quantity;

    if (Quantity == 0) {
      NonExisting.back().insert(Key);
    } else {
      markExistingKey(Key);
    }
    return Quantity;
  }

  bool contains(const key_type &Key) const { return count(Key) != 0; }

  value_type *tryGet(const key_type &Key) {
    auto Iter = Content.find(Key);
    if (Iter == Content.end()) {
      if (TrackingIsActive)
        NonExisting.back().insert(Key);
      return nullptr;
    }

    markExistingKey(Key);
    return &*Iter;
  }

  const value_type *tryGet(const key_type &Key) const {
#ifdef TUPLE_TREE_GENERATOR_EMIT_TRACKING_DEBUG
    onFieldAccess("tryGet", name());
#endif
    auto Iter = Content.find(Key);
    if (Iter == Content.end()) {
      if (TrackingIsActive)
        NonExisting.back().insert(Key);
      return nullptr;
    }

    markExistingKey(Key);
    return &*Iter;
  }

  const_iterator find(const key_type &Key) const {
#ifdef TUPLE_TREE_GENERATOR_EMIT_TRACKING_DEBUG
    onFieldAccess("find", name());
#endif
    Exact.access();
    return Content.find(Key);
  }

  const_iterator lower_bound(const key_type &Key) const {
#ifdef TUPLE_TREE_GENERATOR_EMIT_TRACKING_DEBUG
    onFieldAccess("lower_bound", name());
#endif
    auto Iter = Content.lower_bound(Key);
    Exact.access();
    return Iter;
  }

  const_iterator upper_bound(const key_type &Key) const {
#ifdef TUPLE_TREE_GENERATOR_EMIT_TRACKING_DEBUG
    onFieldAccess("upper_bound", name());
#endif
    auto Iter = Content.upper_bound(Key);
    Exact.access();
    return Iter;
  }

  /// This method lets you use the underlying container as if it wasn't wrapped.
  ///
  /// \note only use this if you know what you're doing, any other alternative
  ///       is preferable if available.
  ///
  /// \note this marks the entire container as read.
  ///
  /// \note non-const version is not provided intentionally, hopefully we never
  ///       need it.
  T const &unwrap() const {
#ifdef TUPLE_TREE_GENERATOR_EMIT_TRACKING_DEBUG
    onFieldAccess("unwrap", name());
#endif
    Exact.access();
    return Content;
  }
  /// @}

public:
  using BatchInserter = typename T::BatchInserter;

  BatchInserter batch_insert() { return BatchInserter(Content); }

  using BatchInsertOrAssigner = typename T::BatchInsertOrAssigner;

  BatchInsertOrAssigner batch_insert_or_assign() {
    return Content.batch_insert_or_assign();
  }

  /// \note This function should always return true
  bool isSorted() const { return Content.isSorted(); }

  /// Allows for partial iteration of the underlying container in the cases
  /// where it's just a vector in disguise.
  ///
  /// \note: this method mercilessly asserts if keys are not correctly ordered
  ///        DO NOT use this for anything that's not verified to have ordered
  ///        keys (like `CFT::Arguments`).
  cppcoro::generator<const_reference> asVector() const
    requires(std::unsigned_integral<key_type>)
  {
    std::remove_const_t<key_type> Index = 0;
    while (Index < Content.size()) {
      const_pointer MaybeElement = tryGet(Index++);
      revng_assert(MaybeElement != nullptr);
      co_yield *MaybeElement;
    }

    revng_assert(Index == Content.size());
    co_return;
  }

private:
  void markExistingKey(const key_type &Key) const {
    if (not TrackingIsActive)
      return;
    auto Iter = Content.find(Key);
    revng_assert(Iter != Content.end());
    revng_assert(Existing.back().size() == Content.size());

    Existing.back()[std::distance(Content.begin(), Iter)] = true;
  }

  TrackingSet getNonExistingRequestedKeys() const {
    TrackingSet Set;
    for (auto &Stack : NonExisting)
      llvm::copy(Stack, std::inserter(Set, Set.begin()));

    return Set;
  }

  TrackingSet getExistingRequestedKeys() const {
    TrackingSet Set;
    for (auto &BitVector : Existing) {
      revng_assert(Content.size() == BitVector.size());
      for (size_t I = 0; I < Content.size(); I++) {
        if (not BitVector[I])
          continue;
        using KOT = KeyedObjectTraits<value_type>;
        Set.insert(KOT::key(*std::next(Content.begin(), I)));
      }
    }

    return Set;
  }

  static auto name() { return "Container<" + valueName().str() + ">"; }

  static auto valueName() -> llvm::StringRef {

    if constexpr (SpecializationOf<typename T::value_type, UpcastablePointer>) {
      using value_type = typename T::value_type::element_type;
      if constexpr (detail::HasName<value_type>)
        return TupleLikeTraits<value_type>::Name;
    } else {
      using value_type = typename T::value_type;
      if constexpr (detail::HasName<value_type>)
        return TupleLikeTraits<value_type>::Name;
    }

    return "unknown";
  }
};

} // namespace revng

template<typename T>
using TrackingSortedVector = revng::TrackingContainer<SortedVector<T>>;

template<typename T>
using TrackingMutableSet = revng::TrackingContainer<MutableSet<T>>;
