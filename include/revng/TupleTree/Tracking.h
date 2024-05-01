#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <set>

#include "revng/TupleTree/TupleTreePath.h"

/// Struct returned by Tracking::collect.
/// The field Read is the set of paths of fields that were accessed.
/// The exact vectors contains the paths of all vectors that were marked as
/// requiring being identical.
///
/// We divide into read and exact vectors because when we will introduce the
/// cold start invalidation based on the hash, the hash will be different
/// depending if they are just read or if they are vectors required to be exact.
struct ReadFields {
  std::set<TupleTreePath> Read;
  std::set<TupleTreePath> ExactVectors;
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
