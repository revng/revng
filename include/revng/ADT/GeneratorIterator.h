#pragma once

//
// Copyright rev.ng Labs Srl. See LICENSE.md for details.
//

#include <cstddef>
#include <optional>
#include <type_traits>

#include "revng/Support/Debug.h"
#include "revng/Support/Generator.h"
#include "revng/Support/IRHelpers.h"

// Define a custom concept for restricting the T type
template<typename T>
concept TriviallyCopyable = std::is_trivially_copyable_v<T>
                            and std::is_copy_constructible_v<T>
                            and std::is_copy_assignable_v<T>;

// Iterator class that generates the next element on increment, holding a
// cppcoro::generator.
//
// We restrict the `T` type so that is a trivially copyable object
template<TriviallyCopyable T>
class GeneratorIterator {
public:
  // We need the specification of the following `iterator_traits` so that
  // `llvm::GraphWriter` doesn't complain in trying to perform `std::distance`
  // over the `child_begin`/`child_end` traits
  using difference_type = ssize_t;
  using value_type = T;
  using pointer = T *;
  using reference = T &;
  using iterator_category = typename std::forward_iterator_tag;

  using inner_iterator = decltype(cppcoro::generator<T>().begin());

private:
  mutable cppcoro::generator<T> Generator;
  mutable inner_iterator Begin;

  // Optional field is used when constructing a snapshotted iterator, i.e. a
  // special iterator state which can only be dereferenced.
  mutable std::optional<T> Snapshot;

public:
  // Constructor building an empty coroutine, with no snapshots.
  // This is basically a coroutine that is already ended, representing an empty
  // range.
  GeneratorIterator() : Generator(), Begin(), Snapshot(std::nullopt) {}

  // Standard constructor providing an input generator, that will be used under
  // the hood to generate the iterators.
  GeneratorIterator(cppcoro::generator<T> &&TheGenerator) :
    Generator(std::move(TheGenerator)),
    Begin(Generator.begin()),
    Snapshot(std::nullopt) {}

  ~GeneratorIterator() = default;

private:
  // Constructor used for a snapshotted iterator.
  // This constructor is private since it's only invoked in the postincrement
  // operator, and doesn't really make sense as public API.
  GeneratorIterator(const T &Value) : Generator(), Begin(), Snapshot(Value) {}

public:
  // Copy assignment.
  GeneratorIterator &operator=(const GeneratorIterator &Other) {
    if (this != &Other) {
      this->Snapshot = Other.Snapshot;

      // We move the coroutine, even if we're in the copy constructor, because
      // the coroutine is not copyable.
      // In principle we wouldn't like to define this copy constructor at all,
      // but there's come code in LLVM we depend upon that requires it and that
      // we don't control.
      // So we have to make sure that the coroutine is never copied and is in
      // fact moved.
      this->Generator = std::move(Other.Generator);
      this->Begin = std::move(Other.Begin);
      Other.Begin = {};

      // In the copy constructor, we want to transition the `Other` object to
      // the snapshot state, if it is not already a snapshot, and if the
      // internal generator iterator is in a valid state (not at the end). If
      // that's the case, we formally leave `Other` as a non snapshot
      // `GeneratorIterator`, but that cannot be dereferenced, which reflects
      // the semantics of trying to copy a `GeneratorIterator` that cannot be
      // dereferenced itself.
      if (not Other.Snapshot and this->Begin != this->Generator.end()) {
        Other.Snapshot = *this->Begin;
      }
    }
    return *this;
  }

  // Copy constructor which uses the copy assignment
  GeneratorIterator(const GeneratorIterator &Other) { *this = Other; }

  // Move assignment
  GeneratorIterator &operator=(GeneratorIterator &&Other) {
    if (this != &Other) {
      this->Snapshot = std::move(Other.Snapshot);
      this->Generator = std::move(Other.Generator);
      this->Begin = std::move(Other.Begin);
    }
    return *this;
  }

  // Move constructor which uses the move assignment
  GeneratorIterator(GeneratorIterator &Other) { *this = std::move(Other); }

public:
  bool isSnapshot() const { return Snapshot.has_value(); }

public:
  bool operator==(const GeneratorIterator &Other) const {
    // It's not legal to compare iterators in snapshot state.
    revng_assert(not this->isSnapshot() and not Other.isSnapshot());

    return this->Begin == Other.Begin;
  }

  GeneratorIterator &operator++() {
    // It's not legal to increment an iterator in snapshot state.
    revng_assert(not isSnapshot());
    ++Begin;
    return *this;
  }

  GeneratorIterator operator++(int) {
    // It's not legal to increment an iterator in snapshot state.
    revng_assert(not isSnapshot());

    // We return an iterator in the snapshot state, that can only be
    // dereferenced to obtain the snapshotted value.
    GeneratorIterator TakenSnapshot(*Begin);
    ++*this;
    return TakenSnapshot;
  }

  T operator*() const {
    if (Snapshot.has_value()) {
      return Snapshot.value();
    } else {
      return *Begin;
    }
  }
};
