#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <set>

#include "revng/Pipeline/DynamicHierarchy.h"
#include "revng/Pipeline/PathComponent.h"
#include "revng/Support/Assert.h"

namespace pipeline {
class Target;
class Context;

/// The granularity tree is a tree used by targets to find out how many names
/// are required to name a target
class Rank : public DynamicHierarchy<Rank> {
public:
  Rank(llvm::StringRef Name) : DynamicHierarchy(Name) {}
  Rank(llvm::StringRef Name, Rank &Parent) :
    DynamicHierarchy<Rank>(Name, Parent) {}
};

class TargetsList;

/// A Kind is used to accumunate objects that logically belongs to the same
/// cathegory.
///
/// As an example, the Kind of Isolated Functions, the Kind of translated
/// binaries.
///
/// A kind refers to a granularity which is used by Targets, furthermore a Kind
/// can optionally have a Parent Kind. Again this is used mostly by targets.
///
/// If you wish to declare a kind to be used in a llvm container, declare a
/// LLVMKind instead.
class Kind : public DynamicHierarchy<Kind> {
private:
  const Rank *TheRank;

public:
  Kind(llvm::StringRef Name, const Rank *TheRank) :
    DynamicHierarchy<Kind>(Name), TheRank(TheRank) {
    revng_assert(TheRank != nullptr);
  }
  Kind(llvm::StringRef Name, Kind &Parent, const Rank *TheRank) :
    DynamicHierarchy<Kind>(Name, Parent), TheRank(TheRank) {
    revng_assert(TheRank != nullptr);
  }

public:
  size_t depth() const { return TheRank->depth(); }

  const Rank &granularity() const { return *TheRank; }

public:
  virtual ~Kind() = default;

  /// A Kind can have a  expand target method, which will be used to expand all
  /// targets that rapresents a collection of objects rather than a scalar one.
  ///
  /// As an example, the espansion of Input equal to /Root/* of isolated
  /// functions kind will create a target equivalent to /Root/FX for each FX
  /// container in the revng model.
  virtual void expandTarget(const Context &Ctx,
                            const Target &Input,
                            TargetsList &Output) const;
};

} // namespace pipeline
