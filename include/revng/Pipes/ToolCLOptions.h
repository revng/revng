#pragma once

/*
 * This file is distributed under the MIT License. See LICENSE.mit for details.
 */

#include <string>

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/PluginLoader.h"

#include "revng/Pipes/PipelineManager.h"
#include "revng/Storage/CLPathOpt.h"

namespace revng::pipes {
class ToolCLOptions {
private:
  llvm::cl::list<std::string> InputPipeline;
  InputPathOpt ModelOverride;
  llvm::cl::list<std::string> EnablingFlags;
  llvm::cl::opt<std::string> ExecutionDirectory;
  llvm::cl::alias A1;
  llvm::cl::alias A2;

public:
  ToolCLOptions(llvm::cl::OptionCategory &Category) :

    InputPipeline("P", llvm::cl::desc("<Pipeline>"), llvm::cl::cat(Category)),
    ModelOverride("model",
                  llvm::cl::desc("Load the model from a provided file"),
                  llvm::cl::cat(Category)),
    EnablingFlags("f",
                  llvm::cl::desc("list of pipeline enabling flags"),
                  llvm::cl::cat(Category)),
    ExecutionDirectory("resume",
                       llvm::cl::desc("Directory from which all containers "
                                      "will be loaded before everything else "
                                      "and to which it will be store after "
                                      "everything else"),
                       llvm::cl::cat(Category)),
    A1("l",
       llvm::cl::desc("Alias for --load"),
       llvm::cl::aliasopt(llvm::LoadOpt),
       llvm::cl::cat(Category)),
    A2("m",
       llvm::cl::desc("Alias for --model"),
       llvm::cl::aliasopt(ModelOverride),
       llvm::cl::cat(Category)) {}

  llvm::Error overrideModel(revng::FilePath ModelOverride,
                            PipelineManager &Manager) {
    const auto &Name = ModelGlobalName;
    auto *Model(cantFail(Manager.context().getGlobal<ModelGlobal>(Name)));
    return Model->load(ModelOverride);
  }

  llvm::Expected<revng::pipes::PipelineManager> makeManager() {
    auto Manager = revng::pipes::PipelineManager::create(InputPipeline,
                                                         EnablingFlags,
                                                         ExecutionDirectory);
    if (not Manager)
      return Manager;

    if (ModelOverride.hasValue())
      if (auto Err = overrideModel(*ModelOverride, *Manager); Err)
        return std::move(Err);

    return Manager;
  }
};
} // namespace revng::pipes
