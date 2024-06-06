#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/ADT/ArrayRef.h"

#include "revng/Pipeline/Invokable.h"
#include "revng/Pipeline/Pipe.h"

namespace pipeline {

template<typename Analysis>
class AnalysisWrapperImpl;

class AnalysisWrapperBase : public InvokableWrapperBase {
private:
  std::string BoundName;

public:
  template<typename Analysis>
  using ImplType = AnalysisWrapperImpl<Analysis>;

public:
  virtual ~AnalysisWrapperBase() override = default;

  virtual llvm::ArrayRef<Kind *>
  getAcceptedKinds(size_t ContainerIndex) const = 0;

  virtual std::unique_ptr<AnalysisWrapperBase>
  clone(std::vector<std::string> NewRunningContainersNames = {}) const = 0;

  void invalidate(const GlobalTupleTreeDiff &Diff,
                  ContainerToTargetsMap &Map,
                  const ContainerSet &Containers) const override {
    revng_abort();
  }

  const std::string &getUserBoundName() const { return BoundName; }

  void setUserBoundName(std::string NewName) { BoundName = std::move(NewName); }
};

template<typename Analysis>
class AnalysisWrapperImpl : public AnalysisWrapperBase {
private:
  InvokableWrapperImpl<Analysis> Invokable;

public:
  AnalysisWrapperImpl(Analysis ActualPipe,
                      std::vector<std::string> RunningContainersNames) :
    Invokable(std::move(ActualPipe), std::move(RunningContainersNames)) {}

  AnalysisWrapperImpl(AnalysisWrapperImpl ActualPipe,
                      std::vector<std::string> RunningContainersNames) :
    Invokable(std::move(ActualPipe.Invokable),
              std::move(RunningContainersNames)) {}

  ~AnalysisWrapperImpl() override = default;

public:
  std::unique_ptr<AnalysisWrapperBase>
  clone(std::vector<std::string> NewContainersNames = {}) const override {
    if (NewContainersNames.empty())
      return std::make_unique<AnalysisWrapperImpl>(*this);
    return std::make_unique<AnalysisWrapperImpl>(*this,
                                                 std::move(NewContainersNames));
  }

  llvm::ArrayRef<Kind *>
  getAcceptedKinds(size_t ContainerIndex) const override {
    return Invokable.getPipe().AcceptedKinds.at(ContainerIndex);
  }

  void dump(std::ostream &OS, size_t Indentation) const override {
    Invokable.dump(OS, Indentation);
  }

  void print(const Context &Ctx,
             llvm::raw_ostream &OS,
             size_t Indentation) const override {
    Invokable.print(Ctx, OS, Indentation);
  }

  llvm::Error run(ExecutionContext &Ctx,
                  ContainerSet &Containers,
                  const llvm::StringMap<std::string> &ExtraArgs) override {
    return Invokable.run(Ctx, Containers, ExtraArgs);
  }

  std::vector<std::string> getOptionsNames() const override {
    return Invokable.getOptionsNames();
  }

  std::vector<std::string> getOptionsTypes() const override {
    return Invokable.getOptionsTypes();
  }

  std::vector<std::string> getRunningContainersNames() const override {
    return Invokable.getRunningContainersNames();
  }
  bool isContainerArgumentConst(size_t ArgumentIndex) const override {
    return Invokable.isContainerArgumentConst(ArgumentIndex);
  }
  std::string getName() const override { return Invokable.getName(); }
};

using AnalysisWrapper = InvokableWrapper<AnalysisWrapperBase>;

} // namespace pipeline
