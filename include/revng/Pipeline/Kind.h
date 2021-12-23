#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <set>

#include "revng/Pipeline/DynamicHierarchy.h"
#include "revng/Pipeline/PathComponent.h"
#include "revng/Support/Assert.h"

namespace Pipeline {
class Target;
class Context;

/// The granularity tree is a tree used by targets to
/// find out how many names are required to name a target.
class Granularity : public DynamicHierarchy<Granularity> {
public:
  Granularity(llvm::StringRef Name) : DynamicHierarchy(Name) {}
  Granularity(llvm::StringRef Name, Granularity &Parent) :
    DynamicHierarchy<Granularity>(Name, Parent) {}
};

class TargetsList;

/// A kind is used to accumunate objects that logically belongs to the same
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
///
class Kind : public DynamicHierarchy<Kind> {
private:
  const Granularity *TheGranularity;

public:
  Kind(llvm::StringRef Name, const Granularity *TheGranularity) :
    DynamicHierarchy<Kind>(Name), TheGranularity(TheGranularity) {
    revng_assert(TheGranularity != nullptr);
  }
  Kind(llvm::StringRef Name, Kind &Parent, const Granularity *TheGranularity) :
    DynamicHierarchy<Kind>(Name, Parent), TheGranularity(TheGranularity) {
    revng_assert(TheGranularity != nullptr);
  }

  size_t depth() const { return TheGranularity->depth(); }

  const Granularity &granularity() const { return *TheGranularity; }

  /// A Kind can have a  expand target method, which will be used to expand all
  /// targets that rapresents a collection of objects rather than a scalar one.
  ///
  /// As an example, the espansion of Input equal to /Root/* of isolated
  /// functions kind will create a target equivalent to /Root/FX for each FX
  /// container in the revng model.
  virtual void expandTarget(const Context &Ctx,
                            const Target &Input,
                            TargetsList &Output) const;

  virtual ~Kind() = default;
};

} // namespace Pipeline
