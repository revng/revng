/// \file LLVMPipe.cpp
/// A llvm pipe is a pipe that operates on a llvm container.

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/PassRegistry.h"
#include "llvm/Passes/PassBuilder.h"

#include "revng/Pipeline/GenericLLVMPipe.h"
#include "revng/Pipeline/LLVMContainer.h"
#include "revng/Support/IRHelpers.h"

using namespace std;
using namespace llvm;
using namespace pipeline;
using namespace cl;

void O2Pipe::registerPasses(llvm::legacy::PassManager &Manager) {
  StringMap<llvm::cl::Option *> &Options(getRegisteredOptions());
  getOption<bool>(Options, "disable-machine-licm")->setInitialValue(true);

  PassBuilder Builder;
  Builder.buildPerModuleDefaultPipeline(OptimizationLevel::O2);
}

std::unique_ptr<LLVMPassWrapperBase> PureLLVMPassWrapper::clone() const {
  return std::make_unique<PureLLVMPassWrapper>(*this);
}

void GenericLLVMPipe::run(const ExecutionContext &, LLVMContainer &Container) {
  llvm::legacy::PassManager Manager;
  for (const auto &Element : Passes)
    Element->registerPasses(Manager);
  Manager.run(Container.getModule());
}

void PureLLVMPassWrapper::registerPasses(llvm::legacy::PassManager &Manager) {
  auto *Registry = llvm::PassRegistry::getPassRegistry();
  Manager.add(Registry->getPassInfo(PassName)->createPass());
}
