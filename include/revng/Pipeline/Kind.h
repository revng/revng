#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <set>

#include "revng/Pipeline/GlobalTupleTreeDiff.h"
#include "revng/Pipeline/PathComponent.h"
#include "revng/Pipeline/Rank.h"
#include "revng/Pipeline/RegisterKind.h"
#include "revng/Support/Assert.h"

namespace pipeline {

class Target;
class Context;
class TargetsList;
class ContainerBase;
class InvalidationEventBase;

/// A Kind is used to accumunate objects that logically belongs to the same
/// category.
///
/// As an example, the Kind of Isolated Functions, the Kind of translated
/// binaries.
///
/// A kind refers to a rank which is used by Targets, furthermore a Kind
/// can optionally have a Parent Kind. Again this is used mostly by targets.
///
/// If you wish to declare a kind to be used in a llvm container, declare a
/// LLVMKind instead.
class Kind : public DynamicHierarchy<Kind> {
private:
  RegisterKind Register;
  const Rank *TheRank;

public:
  Kind(llvm::StringRef Name, const Rank *TheRank) :
    DynamicHierarchy<Kind>(Name), Register(*this), TheRank(TheRank) {
    revng_assert(TheRank != nullptr);
  }

  Kind(llvm::StringRef Name, Kind &Parent, const Rank *TheRank) :
    DynamicHierarchy<Kind>(Name, Parent), Register(*this), TheRank(TheRank) {
    revng_assert(TheRank != nullptr);
  }

  /// Kinds may provide a override for this method. They must return success if
  /// ToVerify (which always has this kind as a kind), well formed inside
  /// Container (which always contains ToVerify).
  virtual llvm::Error
  verify(const ContainerBase &Container, const Target &ToVerify) const;

public:
  size_t depth() const { return TheRank->depth(); }

  const Rank &rank() const { return *TheRank; }

public:
  virtual ~Kind() = default;

  /// A Kind can have an expand target method, which will be used to expand all
  /// targets that represents a collection of objects instead of a single one.
  ///
  /// As an example, the expansion of Input equal to /Root/* of isolated
  /// functions kind will create a target equivalent to /Root/FX for each FX
  /// container in the revng model.
  virtual void expandTarget(const Context &Ctx,
                            const Target &Input,
                            TargetsList &Output) const;

  virtual void getInvalidations(pipeline::TargetsList &ToRemove,
                                const GlobalTupleTreeDiff &Diff) const {}

public:
  template<Rank *R>
  static Kind &deadKind() {
    static Kind DeadKind("Dead", R);
    return DeadKind;
  }
};

} // namespace pipeline
