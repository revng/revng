// \file Main.cpp

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/DynamicLibrary.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/PluginLoader.h"
#include "llvm/Support/raw_os_ostream.h"

#include "revng/Model/LoadModelPass.h"
#include "revng/Pipeline/AllRegistries.h"
#include "revng/Pipeline/ContainerSet.h"
#include "revng/Pipeline/CopyPipe.h"
#include "revng/Pipeline/GenericLLVMPipe.h"
#include "revng/Pipeline/LLVMContainerFactory.h"
#include "revng/Pipeline/LLVMGlobalKindBase.h"
#include "revng/Pipeline/Loader.h"
#include "revng/Pipeline/Target.h"
#include "revng/Pipes/ModelGlobal.h"
#include "revng/Pipes/PipelineManager.h"
#include "revng/Pipes/ToolCLOptions.h"
#include "revng/Support/InitRevng.h"

using std::string;
using namespace llvm;
using namespace llvm::cl;
using namespace pipeline;
using namespace ::revng::pipes;
using namespace ::revng;

static cl::list<string> Arguments(Positional,
                                  ZeroOrMore,
                                  desc("<ArtifactToProduce> <InputBinary>"),
                                  cat(MainCategory));

static opt<string> Output("o",
                          desc("Output filepath of produced artifact"),
                          cat(MainCategory),
                          init("-"));

static opt<string> SaveModel("save-model",
                             desc("Save the model at the end of the run"),
                             cat(MainCategory),
                             init(""));

static opt<bool> ListArtifacts("list",
                               desc("list all possible targets of artifact and "
                                    "exit"),
                               cat(MainCategory),
                               init(false));

static cl::list<string> AnalysesLists("analyses-list",
                                      desc("Analyses list to run"),
                                      cat(MainCategory));

static ToolCLOptions BaseOptions(MainCategory);

static ExitOnError AbortOnError;

int main(int argc, const char *argv[]) {
  revng::InitRevng X(argc, argv);

  HideUnrelatedOptions(MainCategory);
  ParseCommandLineOptions(argc, argv);

  Registry::runAllInitializationRoutines();

  auto Manager = AbortOnError(BaseOptions.makeManager());

  if (Arguments.size() == 0) {
    for (const auto &Step : Manager.getRunner())
      if (Step.getArtifactsKind() != nullptr)
        dbg << Step.getName().str() << "\n";
    return EXIT_SUCCESS;
  }

  if (Arguments.size() == 1) {
    AbortOnError(createStringError(inconvertibleErrorCode(),
                                   "expected any number of positional "
                                   "arguments different from 1"));
  }

  auto &InputContainer = Manager.getRunner().begin()->containers()["input"];
  AbortOnError(InputContainer.loadFromDisk(Arguments[1]));

  InvalidationMap InvMap;
  for (auto &AnalysesListName : AnalysesLists) {
    if (!Manager.getRunner().hasAnalysesList(AnalysesListName)) {
      return EXIT_FAILURE;
    }

    AnalysesList AL = Manager.getRunner().getAnalysesList(AnalysesListName);
    AbortOnError(Manager.runAnalyses(AL, InvMap));
  }

  if (not Manager.getRunner().containsStep(Arguments[0])) {
    AbortOnError(createStringError(inconvertibleErrorCode(),
                                   "No known artifact named %s.\n Use `revng "
                                   "artifact` with no arguments to list "
                                   "available artifacts",
                                   Arguments[0].c_str()));
  }
  auto &Step = Manager.getRunner().getStep(Arguments[0]);
  auto &Container = *Step.getArtifactsContainer();
  auto ContainerName = Container.first();
  auto *Kind = Step.getArtifactsKind();

  if (ListArtifacts) {
    Manager.recalculateAllPossibleTargets();
    auto &StepState = *Manager.getLastState().find(Step.getName());
    auto State = StepState.second.find(ContainerName)->second.filter(*Kind);

    for (const auto &Entry : State) {
      Entry.dumpPathComponents(dbg);
      dbg << "\n";
    }
    return EXIT_SUCCESS;
  }

  ContainerToTargetsMap Map;
  if (Arguments.size() == 2) {
    Map.add(ContainerName, Kind->allTargets(Manager.context()));
  } else {
    for (llvm::StringRef Argument : llvm::drop_begin(Arguments, 2)) {
      auto SlashRemoved = Argument.drop_front();
      llvm::SmallVector<StringRef, 2> Components;
      SlashRemoved.split(Components, "/");
      Map.add(ContainerName, Target(Components, *Kind));
    }
  }
  AbortOnError(Manager.getRunner().run(Step.getName(), Map));

  AbortOnError(Manager.storeToDisk());

  const TargetsList &Targets = Map.contains(ContainerName) ?
                                 Map.at(ContainerName) :
                                 TargetsList();
  auto Produced = Container.second->cloneFiltered(Targets);
  AbortOnError(Produced->storeToDisk(Output));

  if (not SaveModel.empty()) {
    auto Context = Manager.context();
    const auto &ModelName = revng::ModelGlobalName;
    auto FinalModel = AbortOnError(Context.getGlobal<ModelGlobal>(ModelName));
    AbortOnError(FinalModel->storeToDisk(SaveModel));
  }

  return EXIT_SUCCESS;
}
