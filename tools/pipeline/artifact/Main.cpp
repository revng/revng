// \file Main.cpp

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <iostream>

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

using std::string;
using namespace llvm;
using namespace llvm::cl;
using namespace pipeline;
using namespace ::revng::pipes;
using namespace ::revng;

static cl::list<string> Arguments(Positional,
                                  ZeroOrMore,
                                  desc("<artifact> <binary>"),
                                  cat(MainCategory));

static OutputPathOpt Output("o",
                            desc("Output filepath of produced artifact"),
                            cat(MainCategory),
                            init(revng::PathInit::Dash));

static OutputPathOpt SaveModel("save-model",
                               desc("Save the model at the end of the run"),
                               cat(MainCategory));

static opt<bool> ListArtifacts("list",
                               desc("list all possible targets of artifact and "
                                    "exit"),
                               cat(MainCategory),
                               init(false));

static cl::list<string> AnalysesLists("analyses-list",
                                      desc("Analyses list to run"),
                                      cat(MainCategory));

static opt<bool> Analyze("analyze",
                         desc("Run all available *-initial-auto-analysis"),
                         cat(MainCategory));

static ToolCLOptions BaseOptions(MainCategory);

static ExitOnError AbortOnError;

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

  if (Arguments.size() == 0) {
    std::cout << "USAGE: revng-artifact [options] <artifact> <binary>\n\n";
    std::cout << "<artifact> can be one of:\n\n";

    std::vector<std::pair<std::string, std::string>> Pairs;
    for (auto &Step : Manager.getRunner())
      if (Step.getArtifactsKind() != nullptr)
        Pairs
          .emplace_back(Step.getName().str(),
                        Step.getArtifactsContainer()->second->mimeType().str());

    printStringPair(Pairs);

    return EXIT_SUCCESS;
  }

  if (Analyze && AnalysesLists.getNumOccurrences() > 0) {
    AbortOnError(createStringError(inconvertibleErrorCode(),
                                   "Cannot use --analyze and --analyses-lists "
                                   "together."));
  }

  if (Arguments.size() == 1) {
    AbortOnError(createStringError(inconvertibleErrorCode(),
                                   "Expected any number of positional "
                                   "arguments different from 1"));
  }

  auto &InputContainer = Manager.getRunner().begin()->containers()["input"];
  AbortOnError(InputContainer.load(FilePath::fromLocalStorage(Arguments[1])));

  TargetInStepSet InvMap;
  for (auto &AnalysesListName : AnalysesLists) {
    if (!Manager.getRunner().hasAnalysesList(AnalysesListName)) {
      AbortOnError(createStringError(inconvertibleErrorCode(),
                                     "No known analyses list named "
                                       + AnalysesListName));
    }
  }

  Task T(2, "revng-artifact");
  T.advance("Run analyses", true);

  // Collect analyses lists to run
  SmallVector<StringRef> ListsToRun;
  if (Analyze) {
    // TODO: drop this once we merge revng-c in here
    SmallVector<StringRef> Lists = { "revng-initial-auto-analysis",
                                     "revng-c-initial-auto-analysis" };
    for (StringRef AnalysisListName : Lists) {

      if (not Manager.getRunner().hasAnalysesList(AnalysisListName)) {
        AbortOnError(createStringError(inconvertibleErrorCode(),
                                       "The \"" + AnalysisListName.str()
                                         + "\" analysis list is not "
                                           "available.\n"));
      }

      ListsToRun.push_back(AnalysisListName);
    }
  } else if (AnalysesLists.size() != 0) {
    for (const std::string &AnalysisList : AnalysesLists)
      ListsToRun.push_back(AnalysisList);
  }

  // Run the analyses lists
  Task T2(ListsToRun.size(), "Run analyses lists");
  for (StringRef AnalysesListName : ListsToRun) {
    T2.advance(AnalysesListName, true);
    AnalysesList AL = Manager.getRunner().getAnalysesList(AnalysesListName);
    AbortOnError(Manager.runAnalyses(AL, InvMap));
  }

  T.advance("Produce artifact", true);
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

  AbortOnError(Manager.store());

  const TargetsList &Targets = Map.contains(ContainerName) ?
                                 Map.at(ContainerName) :
                                 TargetsList();
  auto Produced = Container.second->cloneFiltered(Targets);
  AbortOnError(Produced->store(*Output));

  if (SaveModel.hasValue()) {
    auto Context = Manager.context();
    const auto &ModelName = revng::ModelGlobalName;
    auto FinalModel = AbortOnError(Context.getGlobal<ModelGlobal>(ModelName));
    AbortOnError(FinalModel->store(*SaveModel));
  }

  return EXIT_SUCCESS;
}
