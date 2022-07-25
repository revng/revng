#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/ADT/ConstexprString.h"
#include "revng/Support/DynamicHierarchy.h"

namespace pipeline {

/// The rank tree is a tree used by targets to find out how many names
/// are required to name a target
class Rank : public DynamicHierarchy<Rank> {
public:
  Rank(llvm::StringRef Name) : DynamicHierarchy(Name) {}
  Rank(llvm::StringRef Name, Rank &Parent) : DynamicHierarchy(Name, Parent) {}
};

template<ConstexprString Str, typename LocationType, typename ParentRnk>
class RankDefinition : public Rank {
public:
  static constexpr std::string_view RankName = Str;
  using Type = LocationType;
  using Parent = ParentRnk;

  using Definition = RankDefinition<Str, Type, Parent>;

public:
  explicit RankDefinition(Parent &ParentObj) : Rank(RankName, ParentObj) {}
};

// Root rank specialization
template<ConstexprString Str>
class RankDefinition<Str, void, void> : public Rank {
public:
  static constexpr std::string_view RankName = Str;
  using Definition = RankDefinition<Str, void, void>;
  using Type = void;
  using Parent = void;

public:
  explicit RankDefinition() : Rank(RankName) {}
};
template<ConstexprString Str>
using RootRankDefinition = RankDefinition<Str, void, void>;

// Rank template parameter deduction helper.
template<ConstexprString Name, typename Type, typename Parent>
pipeline::RankDefinition<Name, Type, Parent> defineRank(Parent &ParentObject) {
  return pipeline::RankDefinition<Name, Type, Parent>(ParentObject);
}

} // namespace pipeline
