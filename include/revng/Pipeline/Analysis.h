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

  ~AnalysisWrapperImpl() override = default;

public:
  llvm::ArrayRef<Kind *>
  getAcceptedKinds(size_t ContainerIndex) const override {
    return Invokable.getPipe().AcceptedKinds[ContainerIndex];
  }

  void dump(std::ostream &OS, size_t Indentation) const override {
    Invokable.dump(OS, Indentation);
  }

  void print(const Context &Ctx,
             llvm::raw_ostream &OS,
             size_t Indentation) const override {
    Invokable.print(Ctx, OS, Indentation);
  }

  void run(Context &Ctx, ContainerSet &Containers) override {
    Invokable.run(Ctx, Containers);
  }

  std::vector<std::string> getRunningContainersNames() const override {
    return Invokable.getRunningContainersNames();
  }
  std::string getName() const override { return Invokable.getName(); }
};

using AnalysisWrapper = InvokableWrapper<AnalysisWrapperBase>;

} // namespace pipeline
