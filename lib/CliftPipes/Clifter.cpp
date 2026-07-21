//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/CliftPipes/Clifter.h"
#include "revng/Clifter/Clifter.h"
#include "revng/Model/IRHelpers.h"
#include "revng/Model/NameBuilder.h"

namespace revng::pypeline::piperuns {

Clifter::Clifter(const class Model &Model,
                 llvm::StringRef Config,
                 llvm::StringRef DynamicConfig,
                 const LLVMFunctionContainer &Input,
                 CliftFunctionContainer &Output) :
  Binary(*Model.get().get()), Input(Input), Output(Output) {
}

void Clifter::runOnFunction(const model::Function &Function) {
  ObjectID Object(Function.Entry());
  const llvm::Module &Module = Input.getModule(Object);
  const llvm::Function
    &LLVMFunction = getUniqueIsolatedFunction(Module, Function.Entry());

  mlir::MLIRContext *Context = Output.getContext();
  auto ModuleOpObject = clift::makeModule(*Context);

  auto Importer = clift::Clifter::make(ModuleOpObject.get(), Binary);
  Importer->import(&LLVMFunction);

  Output.assign(Object, std::move(ModuleOpObject));
}

} // namespace revng::pypeline::piperuns
