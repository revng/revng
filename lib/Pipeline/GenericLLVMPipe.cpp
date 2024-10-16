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

class UpdateContract : public llvm::ModulePass {
public:
  static char ID;

public:
  Context &TheContext;
  llvm::ArrayRef<ContractGroup> Contract;
  ContainerToTargetsMap &Requested;
  llvm::ArrayRef<std::string> ContainersName;

public:
  UpdateContract(Context &TheContext,
                 llvm::ArrayRef<ContractGroup> Contract,
                 ContainerToTargetsMap &Requested,
                 const std::string &ContainersName) :
    llvm::ModulePass(ID),
    TheContext(TheContext),
    Contract(Contract),
    Requested(Requested),
    ContainersName(ContainersName) {}

public:
  bool runOnModule(llvm::Module &Module) override {
    for (auto &Entry : Contract)
      Entry.deduceResults(TheContext, Requested, { ContainersName });
    return false;
  }
};

template<typename T>
using RP = RegisterPass<T>;

char UpdateContract::ID = '_';

static RP<UpdateContract>
  X("advance-contract", "Advance pipeline contracts", true, false);

void GenericLLVMPipe::run(ExecutionContext &EC, LLVMContainer &Container) {
  llvm::legacy::PassManager Manager;
  Manager.add(new LoadExecutionContextPass(&EC, Container.name()));
  using ElementType = std::unique_ptr<LLVMPassWrapperBase>;
  for (const ElementType &Element : Passes) {
    Element->registerPasses(Manager);
    Manager.add(new UpdateContract(EC.getContext(),
                                   Element->getContract(),
                                   EC.getCurrentRequestedTargets(),
                                   Container.name()));
  }
  Manager.run(Container.getModule());
}

void PureLLVMPassWrapper::registerPasses(llvm::legacy::PassManager &Manager) {
  auto *Registry = llvm::PassRegistry::getPassRegistry();
  Manager.add(Registry->getPassInfo(PassName)->createPass());
}
