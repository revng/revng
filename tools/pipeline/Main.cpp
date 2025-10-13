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

namespace cl = llvm::cl;
using namespace pipeline;
using namespace ::revng::pipes;
using namespace revng;

static cl::list<std::string> ContainerOverrides("i",
                                                cl::desc("Load the target file "
                                                         "in the target "
                                                         "container at the "
                                                         "target step"),
                                                cl::cat(MainCategory));

static OutputPathOpt SaveModel("save-model",
                               cl::desc("Save the model at the end of the run"),
                               cl::cat(MainCategory));

static cl::opt<std::string> ApplyModelDiff("apply-model-diff",
                                           cl::desc("Apply model diff"),
                                           cl::cat(MainCategory),
                                           cl::init(""));

static cl::opt<bool> ProduceAllPossibleTargets("produce-all",
                                               cl::desc("Try producing all "
                                                        "possible targets"),
                                               cl::cat(MainCategory),
                                               cl::init(false));

static cl::opt<bool> ProduceAllPossibleTargetsSingle("produce-all-single",
                                                     cl::desc("Try producing "
                                                              "all possible "
                                                              "targets one "
                                                              "element at the "
                                                              "time"),
                                                     cl::cat(MainCategory),
                                                     cl::init(false));

static cl::opt<bool> InvalidateAll("invalidate-all",
                                   cl::desc("Try invalidating all possible "
                                            "targets after producing them. "
                                            "Used for debug purposes"),
                                   cl::cat(MainCategory),
                                   cl::init(false));

static cl::opt<bool> DumpPipeline("d",
                                  cl::desc("Dump built pipeline, but dont run "
                                           "it"),
                                  cl::cat(MainCategory));

static cl::list<std::string> StoresOverrides("o",
                                             cl::desc("Store the target "
                                                      "container at the target "
                                                      "step in the target "
                                                      "file"),
                                             cl::cat(MainCategory));

static cl::list<std::string> Produce("produce",
                                     cl::desc("comma separated list of targets "
                                              "to be produced in one sweep."),
                                     cl::cat(MainCategory));

static cl::list<std::string> Analyze("analyze",
                                     cl::desc("analyses to be performed."),
                                     cl::cat(MainCategory));

static cl::opt<bool> PrintBuildableTargets("targets",
                                           cl::desc("Prints the target that "
                                                    "can be produced from the "
                                                    "current status and exit"),
                                           cl::cat(MainCategory));

static cl::alias A2("t",
                    cl::desc("Alias for --targets"),
                    cl::aliasopt(PrintBuildableTargets),
                    cl::cat(MainCategory));

static ToolCLOptions BaseOptions(MainCategory);

static llvm::ExitOnError AbortOnError;

static Runner::State
parseProductionRequest(Runner &Pipeline,
                       llvm::ArrayRef<llvm::StringRef> Targets) {
  Runner::State ToProduce;

  const auto &Registry = Pipeline.getKindsRegistry();
  for (const auto &Target : Targets) {
    auto &&[StepName, Rest] = Target.split("/");
    auto &Step = ToProduce[StepName];
    AbortOnError(parseTarget(Pipeline.getContext(), Step, Rest, Registry));
  }

  return ToProduce;
}

static void runAnalysis(Runner &Pipeline, llvm::StringRef Target) {
  const auto &Registry = Pipeline.getKindsRegistry();

  auto &&[Step, Rest] = Target.split("/");
  auto &&[AnalysisName, Rest2] = Rest.split("/");

  ContainerToTargetsMap ToProduce;
  TargetInStepSet Map;
  AbortOnError(parseTarget(Pipeline.getContext(), ToProduce, Rest2, Registry));
  AbortOnError(Pipeline.runAnalysis(AnalysisName, Step, ToProduce, Map));
}

static void runPipeline(Runner &Pipeline) {
  // First run the requested analyses
  {
    llvm::Task T(Analyze.size(), "revng-pipeline analyses");
    for (llvm::StringRef Entry : Analyze) {
      T.advance(Entry, true);
      runAnalysis(Pipeline, Entry);
    }
  }

  // Then produce the requested targets
  llvm::Task T(Produce.size(), "revng-pipeline produce");
  for (llvm::StringRef Entry : Produce) {
    T.advance(Entry, true);
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
    auto Diff = AbortOnError(fromFileOrSTDIN<Type>(ApplyModelDiff));

    auto &Runner = Manager.getRunner();
    TargetInStepSet Map;
    GlobalTupleTreeDiff GlobalDiff(std::move(Diff), ModelGlobalName);
    AbortOnError(Runner.apply(GlobalDiff, Map));
  }

  runPipeline(Manager.getRunner());

  if (ProduceAllPossibleTargets)
    AbortOnError(Manager.produceAllPossibleTargets());
  else if (ProduceAllPossibleTargetsSingle)
    AbortOnError(Manager.produceAllPossibleSingleTargets());

  if (InvalidateAll)
    AbortOnError(Manager.invalidateAllPossibleTargets());

  AbortOnError(Manager.store(StoresOverrides));
  AbortOnError(Manager.store());

  auto MaybeSaveModel = AbortOnError(SaveModel.get());
  if (MaybeSaveModel.has_value()) {
    auto Context = Manager.context();
    const auto &ModelName = revng::ModelGlobalName;
    auto FinalModel = AbortOnError(Context.getGlobal<ModelGlobal>(ModelName));
    AbortOnError(FinalModel->store(*MaybeSaveModel));
  }

  return EXIT_SUCCESS;
}
