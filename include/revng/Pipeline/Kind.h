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

template<RankSpecialization... T>
inline std::vector<const Rank *>
locationsToRanks(std::tuple<const T &...> Locations) {
  std::vector<const Rank *> Return;
  std::apply([&](auto &&...Loc) { (Return.push_back(&Loc), ...); }, Locations);
  return Return;
}

/// A Kind is used to accumulate objects that logically belongs to the same
/// category.
///
/// As an example, the Kind of Isolated Functions, the Kind of translated
/// binaries.
///
/// A kind refers to a rank which is used by Targets, furthermore a Kind
/// can optionally have a Parent Kind. Again this is used mostly by targets.
///
/// Kinds expose two additional bits of metadata:
/// * DefinedLocations: these are Locations (ranks) which define the locations
/// exposed by the objects of this kind, these assume that all the n keys of the
/// target map to the first n components of the location; if there are trailing
/// components all of them are matched
/// * PreferredKinds: these are other kinds that are suggested when looking for
/// forward references, i.e. Given a location reference in this Kind, which
/// Kinds should I look into to find a definition?
///
/// If you wish to declare a kind to be used in a llvm container, declare a
/// LLVMKind instead.
class Kind : public DynamicHierarchy<Kind> {
private:
  RegisterKind Register;
  const Rank *TheRank;
  std::vector<const Rank *> DefinedLocations;
  std::vector<const Kind *> PreferredKinds;

public:
  template<RankSpecialization BaseRank, typename... T>
    requires(RankConvertibleTo<BaseRank, T> and ...)
  Kind(llvm::StringRef Name,
       const BaseRank &TheRank,
       std::tuple<const T &...> Locations,
       std::vector<const Kind *> PreferredKinds) :
    DynamicHierarchy<Kind>(Name),
    Register(*this),
    TheRank(&TheRank),
    DefinedLocations(locationsToRanks(Locations)),
    PreferredKinds(PreferredKinds) {
    this->PreferredKinds.push_back(this);
  }

  template<RankSpecialization BaseRank, typename... T>
    requires(RankConvertibleTo<BaseRank, T> and ...)
  Kind(llvm::StringRef Name,
       Kind &Parent,
       const BaseRank &TheRank,
       std::tuple<const T &...> Locations,
       std::vector<const Kind *> PreferredKinds) :
    DynamicHierarchy<Kind>(Name, Parent),
    Register(*this),
    TheRank(&TheRank),
    DefinedLocations(locationsToRanks(Locations)),
    PreferredKinds(PreferredKinds) {
    this->PreferredKinds.push_back(this);
  }

  /// Kinds may provide a override for this method. They must return success if
  /// ToVerify (which always has this kind as a kind), well formed inside
  /// Container (which always contains ToVerify).
  virtual llvm::Error verify(const ContainerBase &Container,
                             const Target &ToVerify) const;

public:
  size_t depth() const { return TheRank->depth(); }

  const Rank &rank() const { return *TheRank; }

  const llvm::ArrayRef<const Kind *> preferredKinds() const {
    return PreferredKinds;
  }

  const llvm::ArrayRef<const Rank *> definedLocations() const {
    return DefinedLocations;
  }

public:
  virtual ~Kind() = default;

  virtual void appendAllTargets(const Context &Context,
                                TargetsList &Out) const = 0;
  TargetsList allTargets(const Context &Context) const;

public:
  template<RankSpecialization RankDefinitionType>
  static Kind &deadKind(const RankDefinitionType &Rank);
};

class SingleElementKind : public Kind {
  using Kind::Kind;
  void appendAllTargets(const Context &Context,
                        TargetsList &Out) const override;
};

class DeadKind : public Kind {

public:
  template<RankSpecialization BaseRank>
  DeadKind(const BaseRank &R) : Kind("Dead", R, {}, {}) {}

  void appendAllTargets(const Context &Context,
                        TargetsList &Out) const override {
    revng_abort();
  }
};

template<RankSpecialization RankDefinitionType>
inline Kind &Kind::deadKind(const RankDefinitionType &Rank) {
  static DeadKind Kind(Rank);
  return Kind;
}

} // namespace pipeline
