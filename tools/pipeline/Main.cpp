// \file Main.cpp

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <cstdlib>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/DynamicLibrary.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_os_ostream.h"

#include "revng/Model/LoadModelPass.h"
#include "revng/Pipeline/AllRegistries.h"
#include "revng/Pipeline/ContainerSet.h"
#include "revng/Pipeline/CopyPipe.h"
#include "revng/Pipeline/GenericLLVMPipe.h"
#include "revng/Pipeline/LLVMContainerFactory.h"
#include "revng/Pipeline/LLVMKind.h"
#include "revng/Pipeline/Loader.h"
#include "revng/Pipeline/Runner.h"
#include "revng/Pipeline/Target.h"
#include "revng/Pipes/ModelGlobal.h"
#include "revng/Pipes/PipelineManager.h"
#include "revng/Pipes/ToolCLOptions.h"
#include "revng/Support/InitRevng.h"
#include "revng/TupleTree/TupleTreeDiff.h"

using std::string;
using namespace llvm;
using namespace llvm::cl;
using namespace pipeline;
using namespace ::revng::pipes;
using namespace revng;

static cl::list<string> ContainerOverrides("i",
                                           desc("Load the target file in the "
                                                "target container at the "
                                                "target step"),
                                           cat(MainCategory));

static opt<string> SaveModel("save-model",
                             desc("Save the model at the end of the run"),
                             cat(MainCategory),
                             init(""));

static opt<string> ApplyModelDiff("apply-model-diff",
                                  desc("Apply model diff"),
                                  cat(MainCategory),
                                  init(""));

static opt<bool> ProduceAllPossibleTargets("produce-all",
                                           desc("Try producing all possible "
                                                "targets"),
                                           cat(MainCategory),
                                           init(false));

static opt<bool> ProduceAllPossibleTargetsSingle("produce-all-single",
                                                 desc("Try producing all "
                                                      "possible "
                                                      "targets one element at "
                                                      "the time"),
                                                 cat(MainCategory),
                                                 init(false));

static cl::list<string> AnalysesLists("analyses-list",
                                      desc("Analyses list to run"),
                                      cat(MainCategory));

static opt<bool> InvalidateAll("invalidate-all",
                               desc("Try invalidating all possible "
                                    "targets after producing them. Used for "
                                    "debug purposes"),
                               cat(MainCategory),
                               init(false));

static opt<bool> DumpPipeline("d",
                              desc("Dump built pipeline, but dont run it"),
                              cat(MainCategory));

static cl::list<string> StoresOverrides("o",
                                        desc("Store the target container at "
                                             "the "
                                             "target step in the target file"),
                                        cat(MainCategory));

static cl::list<string> Produce("produce",
                                desc("comma separated list of targets to be "
                                     "produced in one sweep."),
                                cat(MainCategory));

static cl::list<string>
  Analyze("analyze", desc("analyses to be performed."), cat(MainCategory));

static opt<bool> PrintBuildableTargets("targets",
                                       desc("Prints the target that can be "
                                            "produced from the current status "
                                            "and exit"),
                                       cat(MainCategory));

static alias A2("t",
                desc("Alias for --targets"),
                aliasopt(PrintBuildableTargets),
                cat(MainCategory));

static ToolCLOptions BaseOptions(MainCategory);

static ExitOnError AbortOnError;

static Runner::State
parseProductionRequest(Runner &Pipeline,
                       llvm::ArrayRef<llvm::StringRef> Targets) {
  Runner::State ToProduce;

  const auto &Registry = Pipeline.getKindsRegistry();
  for (const auto &Target : Targets) {
    auto [StepName, Rest] = Target.split("/");
    auto &Step = ToProduce[StepName];
    AbortOnError(parseTarget(Pipeline.getContext(), Step, Rest, Registry));
  }

  return ToProduce;
}

static void runAnalysis(Runner &Pipeline, llvm::StringRef Target) {
  const auto &Registry = Pipeline.getKindsRegistry();

  auto [Step, Rest] = Target.split("/");
  auto [AnalysisName, Rest2] = Rest.split("/");

  ContainerToTargetsMap ToProduce;
  InvalidationMap Map;
  AbortOnError(parseTarget(Pipeline.getContext(), ToProduce, Rest2, Registry));
  AbortOnError(Pipeline.runAnalysis(AnalysisName, Step, ToProduce, Map));
}

static void runPipeline(Runner &Pipeline) {

  for (llvm::StringRef Entry : Analyze) {
    runAnalysis(Pipeline, Entry);
  }

  for (llvm::StringRef Entry : Produce) {
    llvm::SmallVector<llvm::StringRef, 3> Targets;
    Entry.split(Targets, ",");
    AbortOnError(Pipeline.run(parseProductionRequest(Pipeline, Targets)));
  }
}

int main(int argc, char *argv[]) {
  revng::InitRevng X(argc, argv, "", { &MainCategory });

  Registry::runAllInitializationRoutines();

  auto Manager = AbortOnError(BaseOptions.makeManager());

  for (const auto &Override : ContainerOverrides)
    AbortOnError(Manager.overrideContainer(Override));

  if (DumpPipeline) {
    Manager.dump();
    return EXIT_SUCCESS;
  }
  if (PrintBuildableTargets) {
    llvm::raw_os_ostream OS(dbg);
    Manager.writeAllPossibleTargets(OS);
    return EXIT_SUCCESS;
  }

  if (not ApplyModelDiff.empty()) {
    using Type = TupleTreeDiff<model::Binary>;
    auto Diff = AbortOnError(deserializeFileOrSTDIN<Type>(ApplyModelDiff));

    auto &Runner = Manager.getRunner();
    InvalidationMap Map;
    AbortOnError(Runner.apply(GlobalTupleTreeDiff(std::move(Diff)), Map));
  }

  InvalidationMap InvMap;
  for (auto &AnalysesListName : AnalysesLists) {
    if (!Manager.getRunner().hasAnalysesList(AnalysesListName)) {
      AbortOnError(createStringError(inconvertibleErrorCode(),
                                     "no known analyses list named %s, invoke "
                                     "this command without arguments to see "
                                     "the list of available analysis",
                                     AnalysesListName.c_str()));
    }

    AnalysesList AL = Manager.getRunner().getAnalysesList(AnalysesListName);
    AbortOnError(Manager.runAnalyses(AL, InvMap));
  }

  runPipeline(Manager.getRunner());

  if (ProduceAllPossibleTargets)
    AbortOnError(Manager.produceAllPossibleTargets());
  else if (ProduceAllPossibleTargetsSingle)
    AbortOnError(Manager.produceAllPossibleSingleTargets());

  if (InvalidateAll) {
    AbortOnError(Manager.invalidateAllPossibleTargets());
  }

  AbortOnError(Manager.store(StoresOverrides));
  AbortOnError(Manager.storeToDisk());

  if (not SaveModel.empty()) {
    auto Context = Manager.context();
    const auto &ModelName = revng::ModelGlobalName;
    auto FinalModel = AbortOnError(Context.getGlobal<ModelGlobal>(ModelName));
    AbortOnError(FinalModel->storeToDisk(SaveModel));
  }

  return EXIT_SUCCESS;
}
