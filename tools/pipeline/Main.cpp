// \file Main.cpp

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <cstdlib>

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/DynamicLibrary.h"
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

using std::string;
using namespace llvm;
using namespace llvm::cl;
using namespace pipeline;
using namespace ::revng::pipes;

static Logger<> PipelineLogger("pipeline");

cl::OptionCategory PipelineCategory("revng-pipeline options", "");

static cl::list<string>
  InputPipeline("P", desc("<Pipeline>"), cat(PipelineCategory));

static cl::list<string> Targets(Positional,
                                OneOrMore,
                                desc("<Targets to produce>..."),
                                cat(PipelineCategory));

static cl::list<string> ContainerOverrides("i",
                                           desc("Load the target file in the "
                                                "target container at the "
                                                "target step"),
                                           cat(PipelineCategory));

static opt<string> ModelOverride("m",
                                 desc("Load the model from a provided file"),
                                 cat(PipelineCategory),
                                 init(""));

static opt<string> SaveModel("save-model",
                             desc("Save the model at the end of the run"),
                             cat(PipelineCategory),
                             init(""));

static opt<bool> ProduceAllPossibleTargets("produce-all",
                                           desc("Try producing all possible "
                                                "targets"),
                                           cat(PipelineCategory),
                                           init(false));

static opt<bool> InvalidateAll("invalidate-all",
                               desc("Try invalidating all possible "
                                    "targets after producing them. Used for "
                                    "debug purposes"),
                               cat(PipelineCategory),
                               init(false));

static opt<bool> DumpPipeline("d",
                              desc("Dump built pipeline, but dont run it"),
                              cat(PipelineCategory));

static opt<bool> Verbose("verbose",
                         desc("Print explanation while running"),
                         cat(PipelineCategory),
                         init(false));

static alias VerboseAlias1("v",
                           desc("Alias for --verbose"),
                           aliasopt(Verbose),
                           cat(PipelineCategory));

static cl::list<string> StoresOverrides("o",
                                        desc("Store the target container at "
                                             "the "
                                             "target step in the target file"),
                                        cat(PipelineCategory));

static cl::list<string> EnablingFlags("f",
                                      desc("list of pipeline enabling flags"),
                                      cat(PipelineCategory));

static opt<string> ExecutionDirectory("p",
                                      desc("Directory from which all "
                                           "containers will "
                                           "be loaded before everything else "
                                           "and "
                                           "to which it will be store after "
                                           "everything else"),
                                      cat(PipelineCategory));

static cl::list<string>
  LoadLibraries("load", desc("libraries to open"), cat(PipelineCategory));

static alias A1("l",
                desc("Alias for --load"),
                aliasopt(LoadLibraries),
                cat(PipelineCategory));

static opt<bool> PrintBuildableTargets("targets",
                                       desc("Prints the target that can be "
                                            "produced from the current status "
                                            "and exit"),
                                       cat(PipelineCategory));

static alias A2("t",
                desc("Alias for --targets"),
                aliasopt(PrintBuildableTargets),
                cat(PipelineCategory));

static ExitOnError AbortOnError;

static void runPipelineOnce(Runner &Pipeline, llvm::StringRef Target) {
  bool IsAnalysis = llvm::count(Target, ':') == 4;

  ContainerToTargetsMap ToProduce;

  const auto &Registry = Pipeline.getKindsRegistry();
  auto [StepName, Rest] = Target.split(":");
  auto *Stream = Verbose ? &dbgs() : nullptr;

  if (not IsAnalysis) {
    AbortOnError(parseTarget(ToProduce, Rest, Registry));
    AbortOnError(Pipeline.run(StepName, ToProduce, Stream));
    return;
  }

  auto [AnalysisName, Rest2] = Rest.split(":");
  AbortOnError(parseTarget(ToProduce, Rest2, Registry));
  AbortOnError(Pipeline.runAnalysis(AnalysisName, StepName, ToProduce, Stream));
}

static void runPipeline(Runner &Pipeline) {

  for (const auto &Target : Targets)
    runPipelineOnce(Pipeline, Target);
}

static auto makeManager() {
  return PipelineManager::create(InputPipeline,
                                 EnablingFlags,
                                 ExecutionDirectory);
}

int main(int argc, const char *argv[]) {
  HideUnrelatedOptions(PipelineCategory);
  ParseCommandLineOptions(argc, argv);
  auto LoggerOS = PipelineLogger.getAsLLVMStream();

  std::string Msg;
  for (const auto &Library : LoadLibraries) {
    if (sys::DynamicLibrary::LoadLibraryPermanently(Library.c_str(), &Msg))
      AbortOnError(createStringError(inconvertibleErrorCode(), Msg));
  }

  Registry::runAllInitializationRoutines();

  auto Manager = AbortOnError(makeManager());

  for (const auto &Override : ContainerOverrides)
    AbortOnError(Manager.overrideContainer(Override));

  if (not ModelOverride.empty())
    AbortOnError(Manager.overrideModel(ModelOverride));

  if (DumpPipeline) {
    Manager.dump();
    return EXIT_SUCCESS;
  }
  if (PrintBuildableTargets) {
    llvm::raw_os_ostream OS(dbg);
    Manager.writeAllPossibleTargets(OS);
    return EXIT_SUCCESS;
  }

  if (ProduceAllPossibleTargets)
    PipelineLogger.enable();

  runPipeline(Manager.getRunner());

  if (ProduceAllPossibleTargets)
    AbortOnError(Manager.produceAllPossibleTargets(*LoggerOS));

  if (InvalidateAll) {
    PipelineLogger.enable();
    AbortOnError(Manager.invalidateAllPossibleTargets(*LoggerOS));
  }

  AbortOnError(Manager.store(StoresOverrides));
  AbortOnError(Manager.storeToDisk());

  if (not SaveModel.empty()) {
    auto Context = Manager.context();
    const auto &ModelName = ModelGlobalName;
    auto FinalModel = AbortOnError(Context.getGlobal<ModelGlobal>(ModelName));
    AbortOnError(FinalModel->storeToDisk(SaveModel));
  }

  return EXIT_SUCCESS;
}
