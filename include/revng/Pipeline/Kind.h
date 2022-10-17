#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <set>

#include "revng/Pipeline/GlobalTupleTreeDiff.h"
#include "revng/Pipeline/Rank.h"
#include "revng/Pipeline/RegisterKind.h"
#include "revng/Support/Assert.h"

namespace pipeline {

class Target;
class Context;
class TargetsList;
class ContainerBase;
class InvalidationEventBase;

class DeadKind;

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

  virtual void getInvalidations(const Context &Ctx,
                                pipeline::TargetsList &ToRemove,
                                const GlobalTupleTreeDiff &Diff) const {}

  virtual void appendAllTargets(const Context &Ctx, TargetsList &Out) const = 0;
  TargetsList allTargets(const Context &Ctx) const;

public:
  template<typename RankDefinitionType>
  static Kind &deadKind(RankDefinitionType &Rank);
};

class SingleElementKind : public Kind {
  using Kind::Kind;
  void appendAllTargets(const Context &Ctx, TargetsList &Out) const override;
};

class DeadKind : public Kind {

public:
  DeadKind(Rank *R) : Kind("Dead", R) {}
  void appendAllTargets(const Context &Ctx, TargetsList &Out) const override {
    revng_abort();
  }
};

template<typename RankDefinitionType>
inline Kind &Kind::deadKind(RankDefinitionType &Rank) {
  static DeadKind Kind(&Rank);
  return Kind;
}

} // namespace pipeline
