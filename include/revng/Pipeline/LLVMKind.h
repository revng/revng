#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/IR/Function.h"

#include "revng/Pipeline/ContainerEnumerator.h"
#include "revng/Pipeline/ContainerSet.h"
#include "revng/Pipeline/LLVMContainer.h"

namespace pipeline {

/// Kind that must be extended to be able to specify how a to deduce the target
/// associated to a particular llvm global object.
///
/// When extended two function must be provided, symbolToTarget must return the
/// target associated to a function if there is any.
///
/// compactTargets must collapse the targets into the * target if they are all
/// presents, do no thing otherwise.
class LLVMKind : public KindForContainer<LLVMContainer> {
public:
  using StaticContainer = llvm::SmallVector<LLVMKind *, 4>;

public:
  template<RankSpecialization BaseRank>
  LLVMKind(llvm::StringRef Name, const BaseRank &Rank) :
    KindForContainer<LLVMContainer>(Name, Rank) {
    getRegisteredInspectors().push_back(this);
  }

  template<RankSpecialization BaseRank>
  LLVMKind(llvm::StringRef Name, Kind &Parent, const BaseRank &Rank) :
    KindForContainer<LLVMContainer>(Name, Parent, Rank) {
    getRegisteredInspectors().push_back(this);
  }
  ~LLVMKind() override {}

public:
  static bool hasOwner(const llvm::Function &Symbol) {
    for (const auto &I : getRegisteredInspectors())
      if (I->symbolToTarget(Symbol))
        return true;
    return false;
  }

  llvm::Error
  verify(const ContainerBase &Container, const Target &T) const override {
    if (const auto *Casted = llvm::dyn_cast<LLVMContainer>(&Container))
      return verifyTarget(*Casted, T);
    return llvm::Error::success();
  }

  virtual llvm::Error
  verifyTarget(const LLVMContainer &Container, const Target &T) const {
    return llvm::Error::success();
  }

public:
  virtual std::optional<Target>
  symbolToTarget(const llvm::Function &Symbol) const = 0;

public:
  llvm::DenseSet<const llvm::Function *>
  targetsIntersection(const TargetsList &Targets,
                      const LLVMContainer &Container) const {

    llvm::DenseSet<const llvm::Function *> ToReturn;
    for (auto &GL : Container.getModule().functions()) {
      auto MaybeTarget = symbolToTarget(GL);
      if (not MaybeTarget.has_value())
        continue;

      if (not Targets.contains(*MaybeTarget))
        continue;

      ToReturn.insert(&GL);
    }

    return ToReturn;
  }

  llvm::DenseSet<llvm::Function *>
  targetsIntersection(const TargetsList &Targets,
                      LLVMContainer &Container) const {
    llvm::DenseSet<llvm::Function *> ToReturn;
    for (auto &GL : Container.getModule().functions()) {
      auto MaybeTarget = symbolToTarget(GL);
      if (not MaybeTarget.has_value())
        continue;

      if (not Targets.contains(*MaybeTarget))
        continue;

      ToReturn.insert(&GL);
    }

    return ToReturn;
  }

public:
  bool remove(const Context &Ctx,
              const TargetsList &Targets,
              LLVMContainer &Container) const final {

    const bool AllContained = enumerate(Ctx, Container).contains(Targets);

    for (auto &GL : targetsIntersection(Targets, Container))
      GL->deleteBody();

    return AllContained;
  }

  TargetsList
  enumerate(const Context &Ctx, const LLVMContainer &Container) const final {
    TargetsList::List L;
    for (auto &GL : Container.getModule().functions()) {
      auto MaybeTarget = symbolToTarget(GL);
      if (not MaybeTarget.has_value())
        continue;

      L.push_back(std::move(*MaybeTarget));
    }

    return L;
  }

  static llvm::DenseSet<const llvm::Function *>
  untrackedFunctions(const LLVMContainer &Container) {
    llvm::DenseSet<const llvm::Function *> ToReturn;

    for (const auto &F : Container.getModule().functions())
      if (not hasOwner(F))
        ToReturn.insert(&F);

    return ToReturn;
  }

  static llvm::DenseSet<const llvm::Function *>
  functions(const TargetsList &Targets, const LLVMContainer &Container) {
    if (Targets.size() == 0)
      return {};
    llvm::DenseSet<const llvm::Function *> ToReturn;

    for (const auto &Kind : getRegisteredInspectors()) {
      auto Set = Kind->targetsIntersection(Targets, Container);
      for (const auto &E : Set)
        ToReturn.insert(E);
    }

    return ToReturn;
  }

private:
  static StaticContainer &getRegisteredInspectors() {
    static StaticContainer Container;
    return Container;
  }
};

} // namespace pipeline
