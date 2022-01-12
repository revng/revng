/// \file Main.cpp

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <cstdlib>

#include "llvm/Support/DynamicLibrary.h"
#include "llvm/Support/raw_os_ostream.h"

#include "revng/Model/LoadModelPass.h"
#include "revng/Pipeline/ContainerSet.h"
#include "revng/Pipeline/CopyPipe.h"
#include "revng/Pipeline/GenericLLVMPipe.h"
#include "revng/Pipeline/LLVMContainerFactory.h"
#include "revng/Pipeline/LLVMGlobalKindBase.h"
#include "revng/Pipeline/Loader.h"
#include "revng/Pipeline/Registry.h"
#include "revng/Pipeline/Target.h"
#include "revng/Pipes/PipelineManager.h"

using std::string;
using namespace llvm;
using namespace llvm::cl;
using namespace pipeline;
using namespace revng::pipes;

cl::OptionCategory PipelineCategory("revng-pipeline options", "");

static list<string>
  InputPipeline("P", desc("<Pipeline>"), cat(PipelineCategory));

static list<string> Targets(Positional,
                            Required,
                            desc("<Targets to produce>..."),
                            cat(PipelineCategory));

static list<string> ContainerOverrides("i",
                                       desc("Load the target file in the "
                                            "target container at the target "
                                            "step"),
                                       cat(PipelineCategory));

static opt<string> ModelOverride("m",
                                 desc("Load the model from a provided file"),
                                 cat(PipelineCategory),
                                 init(""));

static opt<string> TargetStep("step",
                              desc("name the step in which to produce the "
                                   "elements"),
                              cat(PipelineCategory),
                              init("End"));

static opt<bool> ProduceAllPossibleTargets("produce-all",
                                           desc("Try producing all possible "
                                                "targets"),
                                           cat(PipelineCategory),
                                           init(false));

static opt<bool> DumpPipeline("d",
                              desc("Dump built pipeline and dont run"),
                              cat(PipelineCategory));

static opt<bool> Silence("silent",
                         desc("Do not print explanation while running"),
                         cat(PipelineCategory),
                         init(false));

static alias SilenceAlias1("s",
                           desc("Alias for --silent"),
                           aliasopt(Silence),
                           cat(PipelineCategory));

static list<string> StoresOverrides("o",
                                    desc("Store the target container at the "
                                         "target step in the target file"),
                                    cat(PipelineCategory));

static list<string> EnablingFlags("f",
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

static list<string>
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

static void runPipeline(Runner &Pipeline) {

  ContainerToTargetsMap ToProduce;

  const auto &Registry = Pipeline.getKindsRegistry();
  for (const auto &Target : Targets)
    AbortOnError(parseTarget(ToProduce, Target, Registry));

  auto *Stream = Silence ? nullptr : &dbgs();
  AbortOnError(Pipeline.run(TargetStep, ToProduce, Stream));
}

static auto makeManager() {
  return PipelineManager::create(InputPipeline,
                                 EnablingFlags,
                                 ExecutionDirectory);
}

int main(int argc, const char *argv[]) {
  HideUnrelatedOptions(PipelineCategory);
  ParseCommandLineOptions(argc, argv);

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
    AbortOnError(Manager.printAllPossibleTargets(llvm::dbgs()));
  else
    runPipeline(Manager.getRunner());
  AbortOnError(Manager.store(StoresOverrides));
  AbortOnError(Manager.storeToDisk());

  return EXIT_SUCCESS;
}
