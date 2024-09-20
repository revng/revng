#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <concepts>

#include "llvm/Pass.h"

#include "revng/Model/Function.h"

class LoadModelWrapperPass;

namespace pipeline {
class LoadExecutionContextPass;
class ExecutionContext;
class TargetsList;
} // namespace pipeline

// We shouldn't use the pipeline namespace in revng/Pipes
namespace pipeline {

/// Extent this class to implement a transformation of the IR that needs to read
/// the model and operates function wise.
class FunctionPassImpl {
protected:
  llvm::ModulePass *Pass;

public:
  FunctionPassImpl(llvm::ModulePass &Pass) : Pass(&Pass) {}

  virtual ~FunctionPassImpl() = default;

public:
  virtual bool prologue() { return false; }

  virtual bool runOnFunction(const model::Function &ModelFunction,
                             llvm::Function &Function) = 0;

  virtual bool epilogue() { return false; }

public:
  template<typename T, typename... ArgType>
  T &getAnalysis(ArgType &&...Arg) {
    return Pass->getAnalysis<T>(std::forward<ArgType>(Arg)...);
  }
};

template<typename T>
concept IsFunctionPipeImpl = std::derived_from<std::decay_t<T>,
                                               FunctionPassImpl>;

namespace detail {
bool runOnModule(llvm::Module &Module, FunctionPassImpl &Pipe);
} // namespace detail

/// Wrap your clas deriving from a FunctionPassImpl with this class to turn it
/// compatible with a llvm passmanager
template<IsFunctionPipeImpl T>
class FunctionPass : public llvm::ModulePass {
public:
  static char ID;

public:
  FunctionPass() : llvm::ModulePass::ModulePass(ID) {}

  bool runOnModule(llvm::Module &Module) override {
    T Payload(*this,
              *getAnalysis<LoadModelWrapperPass>()
                 .get()
                 .getReadOnlyModel()
                 .get(),
              Module);
    return detail::runOnModule(Module, Payload);
  }

  void getAnalysisUsage(llvm::AnalysisUsage &AU) const final {
    AU.addRequired<LoadModelWrapperPass>();
    AU.addRequired<pipeline::LoadExecutionContextPass>();
    T::getAnalysisUsage(AU);
  }
};
} // namespace pipeline
