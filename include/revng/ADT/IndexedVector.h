#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <optional>
#include <span>
#include <tuple>
#include <vector>

#include "llvm/ADT/STLExtras.h"

#include "revng/Support/Assert.h"

/// A multimap implemented via an insert-then-use vector.
///
/// This is a insert-then-use data structure. Each inserted element is
/// associated to a key which can be updated with the `changeKey` method.
///
/// Once the insertion is completed, users must call `finalize` and then they
/// can rapidly obtain the range of elements associated to a key.
template<typename Key, typename Value>
class IndexedVector {
private:
  std::vector<std::tuple<Key, size_t, size_t>> Index;
  std::vector<Value> Records;
  bool Finalized = false;
  std::optional<std::pair<Key, size_t>> CurrentKey;

public:
  void changeKey(Key NewKey) {
    revng_assert(not Finalized);
    closeKey();
    CurrentKey.emplace(std::move(NewKey), Records.size());
  }

  void pushBack(Value NewValue) {
    revng_assert(not Finalized);
    revng_assert(CurrentKey.has_value());
    Records.push_back(std::move(NewValue));
  }

  void finalize() {
    revng_assert(not Finalized);
    closeKey();
    llvm::sort(Index);
    Finalized = true;
  }

public:
  std::optional<std::span<const Value>> getRange(Key TheKey) const {
    revng_assert(Finalized);
    auto It = llvm::lower_bound(Index,
                                std::make_tuple(TheKey, size_t(0), size_t(0)));
    if (It == Index.end() or std::get<0>(*It) != TheKey)
      return std::nullopt;
    return std::span<const Value>(&Records[std::get<1>(*It)],
                                  &Records[std::get<2>(*It)]);
  }

private:
  void closeKey() {
    if (CurrentKey.has_value() and CurrentKey->second != Records.size()) {
      Index.emplace_back(std::move(CurrentKey->first),
                         CurrentKey->second,
                         Records.size());
      CurrentKey.reset();
    }
  }
};
