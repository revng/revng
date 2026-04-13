#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/Clift/Helpers.h"
#include "revng/PipeboxCommon/CliftContainers.h"

namespace revng::pypeline::piperuns {

template<typename T>
concept HasRunOnCliftFunctionFunction = requires(T &PipeRun,
                                                 const model::Function
                                                   &Function,
                                                 clift::FunctionOp
                                                   MLIRFunction) {
  { PipeRun.runOnCliftFunction(Function, MLIRFunction) } -> std::same_as<void>;
};

/// Mixin CRTP class that allows a pipe run to define
/// `runOnLLVMFunction(const model::Function &, FunctionOp)` instead of
/// `runOnFunction`, which this class will take care to implement.
template<typename Derived>
class CliftFunctionMixin {
private:
  CliftFunctionContainer &ModuleContainer;

public:
  CliftFunctionMixin(CliftFunctionContainer &ModuleContainer) :
    ModuleContainer(ModuleContainer) {}

  void runOnFunction(const model::Function &Function) {
    static_assert(HasRunOnCliftFunctionFunction<Derived>);
    using namespace clift;
    mlir::ModuleOp Module = ModuleContainer
                              .getModule(ObjectID(Function.Entry()));
    FunctionOp MLIRFunction = getUniqueIsolatedFunction(Module,
                                                        Function.Entry());
    static_cast<Derived *>(this)->runOnCliftFunction(Function, MLIRFunction);
  }
};

} // namespace revng::pypeline::piperuns
