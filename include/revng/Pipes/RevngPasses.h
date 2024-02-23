#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <concepts>

#include "llvm/Pass.h"

#include "revng/Model/Function.h"

namespace pipeline {
class ExecutionContext;
class TargetsList;

/// Class to be extended when a pass withes to have access to both
/// the module and the list of targets to be produced.
class ModulePass : public llvm::ModulePass {
  using llvm::ModulePass::ModulePass;

  bool runOnModule(llvm::Module &Module) final;

  virtual bool run(llvm::Module &Module, const TargetsList &Targets) = 0;
};

/// Extent this class to implement a transformation of the IR that needs to read
/// the model and operates function wise.
class FunctionPassImpl {
private:
  llvm::ModulePass *Pass;

public:
  FunctionPassImpl(llvm::ModulePass &Pass) : Pass(&Pass) {}
  virtual ~FunctionPassImpl() = default;

public:
  virtual bool prologue(llvm::Module &Module, const model::Binary &Binary) = 0;

  virtual bool runOnFunction(llvm::Function &Function,
                             const model::Function &ModelFunction) = 0;

public:
  template<typename T>
  T &getAnalysis() {
    return Pass->getAnalysis<T>();
  }
};

template<typename T>
concept IsFunctionPipeImpl = std::derived_from<std::decay_t<T>,
                                               FunctionPassImpl>;

namespace detail {
bool runOnModule(llvm::Module &Module, FunctionPassImpl &Pipe);
}

/// Wrap your clas deriving from a FunctionPassImpl with this class to turn it
/// compatible with a llvm passmanager
template<IsFunctionPipeImpl T>
class FunctionPass : public llvm::ModulePass {
public:
  FunctionPass(char &ID) : llvm::ModulePass::ModulePass(ID) {}
  bool runOnModule(llvm::Module &Module) override {
    T Payload(*this);
    return detail::runOnModule(Module, Payload);
  }
};
} // namespace pipeline
