#pragma once

/*
 * This file is distributed under the MIT License. See LICENSE.mit for details.
 */

#include "llvm/IR/LegacyPassManager.h"

#include "revng/EarlyFunctionAnalysis/FunctionMetadataCache.h"
#include "revng/Model/LoadModelPass.h"
#include "revng/Pipeline/Context.h"
#include "revng/Pipeline/LLVMContainer.h"
#include "revng/Pipes/ModelGlobal.h"
#include "revng/Support/ResourceFinder.h"

namespace revng::pipes {

template<typename... Passes>
class LLVMAnalysisImplementation {
public:
  void run(const pipeline::ExecutionContext &Ctx,
           pipeline::LLVMContainer &Container) {
    llvm::legacy::PassManager Manager;
    registerPasses(Ctx.getContext(), Manager);
    Manager.run(Container.getModule());
  }

  void print(const pipeline::Context &Ctx,
             llvm::raw_ostream &OS,
             llvm::ArrayRef<std::string> ContainerNames) const {
    OS << *revng::ResourceFinder.findFile("bin/revng");
    OS << " opt --model-path=model.yml " << ContainerNames[0] << " -o "
       << ContainerNames[0];
    std::vector<std::string> PassNames;
    (PassNames.push_back(llvm::PassRegistry::getPassRegistry()
                           ->getPassInfo(&Passes::ID)
                           ->getPassArgument()
                           .str()),
     ...);
    for (const auto &Name : PassNames)
      OS << "-" << Name << " ";
    OS << "\n";
  }

  void registerPasses(const pipeline::Context &Ctx,
                      llvm::legacy::PassManager &Manager) const {
    auto Global = llvm::cantFail(Ctx.getGlobal<ModelGlobal>(ModelGlobalName));
    Manager.add(new LoadModelWrapperPass(ModelWrapper(Global->get())));
    Manager.add(new FunctionMetadataCachePass());
    (Manager.add(new Passes()), ...);
  };
};

} // namespace revng::pipes
