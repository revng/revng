#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/IR/Function.h"

#include "revng/Model/IRHelpers.h"
#include "revng/PipeboxCommon/LLVMContainer.h"

namespace revng::pypeline::piperuns {

template<typename T>
concept HasRunOnLLVMFunctionFunction = requires(T &PipeRun,
                                                const model::Function &Function,
                                                llvm::Function &LLVMFunction) {
  { PipeRun.runOnLLVMFunction(Function, LLVMFunction) } -> std::same_as<void>;
};

/// Mixin CRTP class that allows a pipe run to define
/// `runOnLLVMFunction(const model::Function &, llvm::Function &)` instead of
/// `runOnFunction`, which this class will take care to implement.
template<typename Derived>
class LLVMFunctionMixin {
private:
  revng::pypeline::LLVMFunctionContainer &ModuleContainer;

public:
  LLVMFunctionMixin(revng::pypeline::LLVMFunctionContainer &ModuleContainer) :
    ModuleContainer(ModuleContainer) {}

  void runOnFunction(const model::Function &Function) {
    static_assert(HasRunOnLLVMFunctionFunction<Derived>);
    llvm::Module &Module = ModuleContainer
                             .getModule(ObjectID(Function.Entry()));
    llvm::Function &LLVMFunction = getUniqueIsolatedFunction(Module,
                                                             Function.Entry());
    static_cast<Derived *>(this)->runOnLLVMFunction(Function, LLVMFunction);
  }
};

} // namespace revng::pypeline::piperuns
