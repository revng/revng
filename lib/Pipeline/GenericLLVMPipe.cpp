/// \file LLVMPipe.cpp
/// \brief A llvm pipe is a pipe that operates on a llvm container

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/PassRegistry.h"
#include "llvm/Passes/PassBuilder.h"

#include "revng/Pipeline/GenericLLVMPipe.h"

using namespace std;
using namespace llvm;
using namespace pipeline;
using namespace cl;

void O2Pipe::registerPasses(llvm::legacy::PassManager &Manager) {
  StringMap<llvm::cl::Option *> &Options(getRegisteredOptions());
  getOption<bool>(Options, "disable-machine-licm")->setInitialValue(true);

  PassBuilder Builder;
  Builder.buildPerModuleDefaultPipeline(PassBuilder::OptimizationLevel::O2);
}

std::unique_ptr<LLVMPassWrapperBase> PureLLVMPassWrapper::clone() const {
  return std::make_unique<PureLLVMPassWrapper>(*this);
}
