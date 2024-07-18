#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <set>

#include "revng/ADT/SortedVector.h"
#include "revng/TupleTree/TupleTreePath.h"

// #define USE_SORTEDVECTOR
#define USE_VECTOR

template<>
struct KeyedObjectTraits<TupleTreePath>
  : public IdentityKeyedObjectTraits<TupleTreePath> {};

/// Struct returned by Tracking::collect.
/// The field Read is the set of paths of fields that were accessed.
/// The exact vectors contains the paths of all vectors that were marked as
/// requiring being identical.
///
/// We divide into read and exact vectors because when we will introduce the
/// cold start invalidation based on the hash, the hash will be different
/// depending if they are just read or if they are vectors required to be exact.
struct ReadFields {
#ifdef USE_SORTEDVECTOR
  SortedVector<TupleTreePath> Read;
  SortedVector<TupleTreePath> ExactVectors;
#elif defined(USE_VECTOR)
  template<typename T>
  class Lol : public llvm::SmallVector<T> {
  public:
    using Base = llvm::SmallVector<T>;
    using Base::Base;
    template<typename X>
    void insert(X &&Arg) {
      llvm::SmallVector<T>::push_back(std::forward<X>(Arg));
    }
  };
  Lol<TupleTreePath> Read;
  Lol<TupleTreePath> ExactVectors;
#else
  std::set<TupleTreePath> Read;
  std::set<TupleTreePath> ExactVectors;
#endif
};

namespace revng {

struct Tracking {
  template<typename M>
  static ReadFields collect(const M &LHS);

  template<typename M>
  static void clearAndResume(const M &LHS);

  template<typename M>
  static void push(const M &LHS);

  template<typename M>
  static void pop(const M &LHS);

  template<typename M>
  static void stop(const M &LHS);
};

} // namespace revng
