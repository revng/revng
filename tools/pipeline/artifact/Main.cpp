// \file Main.cpp

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <iostream>
#include <memory>

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/DynamicLibrary.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/PluginLoader.h"
#include "llvm/Support/raw_os_ostream.h"

#include "revng/Model/LoadModelPass.h"
#include "revng/Pipeline/AllRegistries.h"
#include "revng/Pipeline/AnalysesList.h"
#include "revng/Pipeline/ContainerSet.h"
#include "revng/Pipeline/Context.h"
#include "revng/Pipeline/CopyPipe.h"
#include "revng/Pipeline/GenericLLVMPipe.h"
#include "revng/Pipeline/LLVMContainerFactory.h"
#include "revng/Pipeline/LLVMKind.h"
#include "revng/Pipeline/Loader.h"
#include "revng/Pipeline/Target.h"
#include "revng/Pipes/ModelGlobal.h"
#include "revng/Pipes/PipelineManager.h"
#include "revng/Pipes/ToolCLOptions.h"
#include "revng/Support/InitRevng.h"

namespace cl = llvm::cl;
using namespace pipeline;
using namespace ::revng::pipes;
using namespace ::revng;

static cl::list<std::string> Arguments(cl::Positional,
                                       cl::ZeroOrMore,
                                       cl::desc("<artifact> <binary> [TARGET "
                                                "[TARGET [...]]]"),
                                       cl::cat(MainCategory));

static OutputPathOpt Output("o",
                            cl::desc("Output filepath of produced artifact"),
                            cl::cat(MainCategory),
                            cl::init(revng::PathInit::Dash));

static OutputPathOpt SaveModel("save-model",
                               cl::desc("Save the model at the end of the run"),
                               cl::cat(MainCategory));

static cl::opt<bool> ListArtifacts("list",
                                   cl::desc("list all the known artifacts and "
                                            "exit"),
                                   cl::cat(MainCategory),
                                   cl::init(false));

static cl::opt<string> Analyses("analyses",
                                cl::desc("Analyses to run, comma separated"),
                                cl::cat(MainCategory));

static cl::opt<bool> Analyze("analyze",
                             cl::desc("Run revng-initial-auto-analysis"),
                             cl::cat(MainCategory));

static ToolCLOptions BaseOptions(MainCategory);

static llvm::ExitOnError AbortOnError;

inline void
printStringPair(const std::vector<std::pair<std::string, std::string>> &Pairs) {
  unsigned Longest = 0;
  for (auto &[First, Second] : Pairs)
    if (First.size() > Longest)
      Longest = First.size();

  for (auto &[First, Second] : Pairs)
    std::cout << "  " << First << std::string(Longest + 1 - First.size(), ' ')
              << "- " << Second << "\n";
}

int main(int argc, char *argv[]) {
  using revng::FilePath;

  revng::InitRevng X(argc, argv, "", { &MainCategory });

  Registry::runAllInitializationRoutines();

  auto Manager = AbortOnError(BaseOptions.makeManager());
  auto &Runner = Manager.getRunner();

  if (Arguments.size() == 0) {
    std::cout << "USAGE: revng-artifact [options] <artifact> <binary>\n\n";
    std::cout << "<artifact> can be one of:\n\n";

    std::vector<std::pair<std::string, std::string>> Pairs;
    for (auto &Step : Runner)
      if (Step.getArtifactsKind() != nullptr)
        Pairs
          .emplace_back(Step.getName().str(),
                        Step.getArtifactsContainer()->second->mimeType().str());

    printStringPair(Pairs);

    return EXIT_SUCCESS;
  }

  if (not Runner.containsStep(Arguments[0])) {
    AbortOnError(revng::createError("No known artifact named %s.\nUse `revng "
                                    "artifact` with no arguments to list "
                                    "available artifacts.",
                                    Arguments[0].c_str()));
  }

  auto &Step = Runner.getStep(Arguments[0]);
  auto MaybeContainer = Step.getArtifactsContainer();
  if (not MaybeContainer) {
    AbortOnError(revng::createError("The step %s is not associated to "
                                    "an artifact.",
                                    Arguments[0].c_str()));
  }

  if (Analyze && Analyses.getNumOccurrences() > 0) {
    AbortOnError(revng::createError("Cannot use --analyze and --analyses "
                                    "together."));
  }

  if (Arguments.size() == 1) {
    AbortOnError(revng::createError("Expected any number of positional "
                                    "arguments different from 1."));
  }

  auto &InputContainer = Runner.begin()->containers()["input"];
  InputPath = Arguments[1];
  FilePath InputFilePath = FilePath::fromLocalStorage(Arguments[1]);
  AbortOnError(InputFilePath.check());
  AbortOnError(InputContainer.load(InputFilePath));

  llvm::SmallVector<llvm::StringRef> AnalysesToRun;
  if (Analyses.getNumOccurrences() > 0) {
    llvm::StringRef(Analyses.getValue()).split(AnalysesToRun, ',');
    for (llvm::StringRef AnalysesName : AnalysesToRun) {
      if (not Runner.hasAnalysesList(AnalysesName)
          and not Runner.containsAnalysis(AnalysesName)) {
        AbortOnError(revng::createError("No known analysis named "
                                        + AnalysesName + "."));
      }
    }
  }

  llvm::Task T(2, "revng-artifact");
  T.advance("Run analyses", true);

  // Collect analyses lists to run
  if (Analyze) {
    llvm::StringRef List = "revng-initial-auto-analysis";

    if (not Runner.hasAnalysesList(List)) {
      AbortOnError(revng::createError("The \"" + List.str()
                                      + "\" analysis list is not "
                                        "available.\n"));
    }

    AnalysesToRun.insert(AnalysesToRun.begin(), List);
  }

  // Run the analyses lists
  llvm::Task T2(AnalysesToRun.size(), "Run analyses");
  for (llvm::StringRef AnalysesName : AnalysesToRun) {
    T2.advance(AnalysesName, true);

    auto InvMap = revng::pipes::runAnalysisOrAnalysesList(Manager,
                                                          AnalysesName,
                                                          AbortOnError);
  }

  T.advance("Produce artifact", true);

  auto ContainerName = MaybeContainer->first();
  auto *Kind = Step.getArtifactsKind();

  if (ListArtifacts) {
    Manager.recalculateAllPossibleTargets();
    auto &StepState = *Manager.getLastState().find(Step.getName());
    auto State = StepState.second.find(ContainerName)->second.filter(*Kind);

    for (const auto &Entry : State) {
      dbg << Entry.path() << "\n";
    }
    return EXIT_SUCCESS;
  }

  ContainerToTargetsMap Map;
  if (Arguments.size() == 2) {
    Map.add(ContainerName, Kind->allTargets(Manager.context()));
  } else {
    for (llvm::StringRef Argument : llvm::drop_begin(Arguments, 2)) {
      std::string ArgumentWithKind = (Argument + ":" + Kind->name()).str();
      auto
        RequestedTarget = AbortOnError(Target::deserialize(Manager.context(),
                                                           ArgumentWithKind));
      Map.add(ContainerName, RequestedTarget);
    }
  }
  AbortOnError(Runner.run(Step.getName(), Map));

  AbortOnError(Manager.store());

  const TargetsList &Targets = Map.contains(ContainerName) ?
                                 Map.at(ContainerName) :
                                 TargetsList();
  auto Produced = MaybeContainer->second->cloneFiltered(Targets);

  auto MaybeOutput = AbortOnError(Output.get());
  revng_assert(MaybeOutput.has_value());
  AbortOnError(Produced->store(*MaybeOutput));

  auto MaybeSaveModel = AbortOnError(SaveModel.get());
  if (MaybeSaveModel.has_value()) {
    auto Context = Manager.context();
    const auto &ModelName = revng::ModelGlobalName;
    auto FinalModel = AbortOnError(Context.getGlobal<ModelGlobal>(ModelName));
    AbortOnError(FinalModel->store(*MaybeSaveModel));
  }

  return EXIT_SUCCESS;
}
