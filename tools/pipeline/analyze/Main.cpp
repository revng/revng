// \file Main.cpp

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <system_error>

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/DynamicLibrary.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/raw_os_ostream.h"

#include "revng/Model/LoadModelPass.h"
#include "revng/Pipeline/AllRegistries.h"
#include "revng/Pipeline/ContainerSet.h"
#include "revng/Pipeline/CopyPipe.h"
#include "revng/Pipeline/GenericLLVMPipe.h"
#include "revng/Pipeline/Global.h"
#include "revng/Pipeline/LLVMContainerFactory.h"
#include "revng/Pipeline/LLVMGlobalKindBase.h"
#include "revng/Pipeline/Loader.h"
#include "revng/Pipeline/Step.h"
#include "revng/Pipeline/Target.h"
#include "revng/Pipes/ModelGlobal.h"
#include "revng/Pipes/PipelineManager.h"
#include "revng/Pipes/ToolCLOptions.h"
#include "revng/TupleTree/TupleTree.h"

using std::string;
using namespace llvm;
using namespace llvm::cl;
using namespace pipeline;
using namespace ::revng::pipes;

static cl::list<string> Arguments(Positional,
                                  ZeroOrMore,
                                  desc("<ArtifactToProduce> <InputBinary>"),
                                  cat(MainCategory));

static opt<string> Output("o",
                          desc("Output filepath of produced model"),
                          cat(MainCategory),
                          init("-"));

static opt<bool> NoApplyModel("no-apply",
                              desc("run the analysis but do not apply it (used "
                                   "to recreate consistent debug "
                                   "configurations)"),
                              cat(MainCategory),
                              init(false));

static ToolCLOptions BaseOptions(MainCategory);

static ExitOnError AbortOnError;

static TupleTreeGlobal<model::Binary> &getModel(PipelineManager &Manager) {
  auto &Context = Manager.context();
  const auto &ModelName = ModelGlobalName;
  auto *FinalModel = AbortOnError(Context.getGlobal<ModelGlobal>(ModelName));
  revng_assert(FinalModel != nullptr);
  return *FinalModel;
}

static Step *
getStepOfAnalysis(pipeline::Runner &Runner, llvm::StringRef AnalysisName) {
  const auto &StepHasAnalysis = [AnalysisName](const Step &Step) {
    return Step.hasAnalysis(AnalysisName);
  };
  auto It = find_if(Runner, StepHasAnalysis);
  if (It == Runner.end())
    return nullptr;
  return &*It;
}

static llvm::Error
overrideModel(PipelineManager &Manager, TupleTree<model::Binary> NewModel) {
  const auto &Name = ModelGlobalName;
  auto *Model(cantFail(Manager.context().getGlobal<ModelGlobal>(Name)));
  Model->get() = std::move(NewModel);
  return llvm::Error::success();
}

int main(int argc, const char *argv[]) {
  HideUnrelatedOptions(MainCategory);
  ParseCommandLineOptions(argc, argv);

  Registry::runAllInitializationRoutines();

  auto Manager = AbortOnError(BaseOptions.makeManager());
  auto OriginalModel = getModel(Manager);

  if (Arguments.size() == 0) {
    dbg << "all\n";
    for (const auto &Step : Manager.getRunner())
      for (const auto &Analysis : Step.analyses())
        dbg << Analysis.getKey().str() << "\n";
    return EXIT_SUCCESS;
  }

  if (Arguments.size() == 1) {
    AbortOnError(createStringError(inconvertibleErrorCode(),
                                   "expected any number of positional "
                                   "arguments different from 1"));
  }

  auto &InputContainer = Manager.getRunner().begin()->containers()["input"];
  AbortOnError(InputContainer.loadFromDisk(Arguments[1]));

  if (Arguments[0] == "all") {
    AbortOnError(Manager.getRunner().runAllAnalyses());
  } else {
    auto *Step = getStepOfAnalysis(Manager.getRunner(), Arguments[0]);
    if (not Step) {
      AbortOnError(createStringError(inconvertibleErrorCode(),
                                     "no known analysis named %s, invoke this "
                                     "command without arguments to see the "
                                     "list "
                                     "of aviable analysis",
                                     Arguments[0].c_str()));
    }

    auto &Analysis = Step->getAnalysis(Arguments[0]);

    ContainerToTargetsMap Map;
    for (const auto &Pair :
         llvm::enumerate(Analysis->getRunningContainersNames())) {
      auto Index = Pair.index();
      const auto &ContainerName = Pair.value();
      for (const auto &Kind : Analysis->getAcceptedKinds(Index)) {
        Map.add(ContainerName, Target(*Kind));
      }
    }

    AbortOnError(Manager.getRunner().runAnalysis(Analysis->getName(),
                                                 Step->getName(),
                                                 Map));
  }
  if (NoApplyModel)
    AbortOnError(overrideModel(Manager, OriginalModel.get()));

  AbortOnError(Manager.storeToDisk());

  auto &FinalModel = getModel(Manager);
  AbortOnError(FinalModel.storeToDisk(Output));

  return EXIT_SUCCESS;
}
