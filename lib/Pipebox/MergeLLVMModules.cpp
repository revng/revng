//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/Pipebox/MergeLLVMModules.h"

namespace revng::pypeline::piperuns {

void MergeLLVMModules::run() {
  llvm::LLVMContext &Context = Output.getModule().getContext();
  auto OutputModule = std::make_unique<llvm::Module>("output", Context);
  for (const ObjectID &Object : Input.objects()) {
    linkFunctionModules(cloneIntoContext(Input.getModule(Object), Context),
                        OutputModule);
  }
  Output.assign(std::move(OutputModule));
}

} // namespace revng::pypeline::piperuns
