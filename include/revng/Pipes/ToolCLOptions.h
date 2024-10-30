#pragma once

/*
 * This file is distributed under the MIT License. See LICENSE.md for details.
 */

#include <string>

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/PluginLoader.h"

#include "revng/Pipes/PipelineManager.h"
#include "revng/Storage/CLPathOpt.h"
#include "revng/Support/Error.h"

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
      if (auto Error = overrideModel(*ModelOverride, *Manager))
        return Error;

    return Manager;
  }
};

inline pipeline::Step *getStepOfAnalysis(pipeline::Runner &Runner,
                                         llvm::StringRef AnalysisName) {
  using namespace pipeline;
  const auto &StepHasAnalysis = [AnalysisName](const Step &Step) {
    return Step.hasAnalysis(AnalysisName);
  };
  auto It = llvm::find_if(Runner, StepHasAnalysis);
  if (It == Runner.end())
    return nullptr;
  return &*It;
}

inline pipeline::TargetInStepSet
runAnalysisOrAnalysesList(revng::pipes::PipelineManager &Manager,
                          llvm::StringRef Name,
                          llvm::ExitOnError &AbortOnError) {
  using namespace pipeline;
  using namespace llvm;
  auto &Runner = Manager.getRunner();
  TargetInStepSet InvMap;
  if (Runner.hasAnalysesList(Name)) {
    AnalysesList AL = Runner.getAnalysesList(Name);
    AbortOnError(Manager.runAnalyses(AL, InvMap));
  } else {
    auto *Step = getStepOfAnalysis(Runner, Name);
    if (not Step) {
      AbortOnError(revng::createError("No known analysis named %s, invoke this "
                                      "command without arguments to see the "
                                      "list of available analysis",
                                      Name.data()));
    }

    auto &Analysis = Step->getAnalysis(Name);

    ContainerToTargetsMap Map;
    for (const auto &Pair :
         llvm::enumerate(Analysis->getRunningContainersNames())) {
      auto Index = Pair.index();
      const auto &ContainerName = Pair.value();
      for (const auto &Kind : Analysis->getAcceptedKinds(Index)) {
        Map.add(ContainerName, Kind->allTargets(Manager.context()));
      }
    }

    llvm::StringRef StepName = Step->getName();
    std::string AnalysisName = Name.str();
    AbortOnError(Manager.runAnalysis(AnalysisName, StepName, Map, InvMap));
  }

  return InvMap;
}

} // namespace revng::pipes
