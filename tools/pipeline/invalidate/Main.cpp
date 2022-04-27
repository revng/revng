/// \file Main.cpp

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <cstdlib>

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

cl::OptionCategory PipelineCategory("revng-invalidate options", "");

static cl::list<string>
  InputPipeline("P", desc("<Pipeline>"), cat(PipelineCategory));

static cl::list<string> Targets(Positional,
                                Required,
                                desc("<Targets to invalidate>..."),
                                cat(PipelineCategory));

static opt<string> TargetStep("step",
                              Required,
                              desc("name the step in which to produce the "
                                   "elements"),
                              cat(PipelineCategory));

static opt<string> ExecutionDirectory("p",
                                      desc("Directory from which all "
                                           "containers will "
                                           "be loaded before everything else "
                                           "and "
                                           "to which it will be store after "
                                           "everything else"),
                                      cat(PipelineCategory),
                                      init("."));

static cl::list<string>
  LoadLibraries("load", desc("libraries to open"), cat(PipelineCategory));

static cl::list<string> EnablingFlags("f",
                                      desc("list of pipeline enabling flags"),
                                      cat(PipelineCategory));

static opt<bool> DumpPredictedRemovals("dump-invalidations",
                                       desc("dump predicted invalidate "
                                            "targets"),
                                       cat(PipelineCategory));

static opt<bool> DumpFinalStatus("dump-status",
                                 desc("dump status after invalidation "
                                      "targets"),
                                 cat(PipelineCategory));

static alias A1("l",
                desc("Alias for --load"),
                aliasopt(LoadLibraries),
                cat(PipelineCategory));

static ExitOnError AbortOnError;

static auto makeManager() {
  return PipelineManager::create(InputPipeline,
                                 EnablingFlags,
                                 ExecutionDirectory);
}

static InvalidationMap getInvalidationMap(Runner &Pipeline) {
  InvalidationMap Invalidations;
  auto &ToInvalidate = Invalidations[TargetStep];

  const auto &Registry = Pipeline.getKindsRegistry();
  for (const auto &Target : Targets)
    AbortOnError(parseTarget(ToInvalidate, Target, Registry));

  return Invalidations;
}

static void
dumpInvalidationMap(llvm::raw_ostream &OS, const InvalidationMap &Map) {
  for (const auto &Pair : Map) {
    OS << Pair.first();
    Pair.second.dump(OS, 1);
  }
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

  auto Map = getInvalidationMap(Manager.getRunner());
  AbortOnError(Manager.getRunner().getInvalidations(Map));

  if (DumpPredictedRemovals) {
    dumpInvalidationMap(llvm::outs(), Map);
    return EXIT_SUCCESS;
  }

  AbortOnError(Manager.getRunner().invalidate(Map));

  if (DumpFinalStatus)
    Manager.dump();

  AbortOnError(Manager.storeToDisk());
  return EXIT_SUCCESS;
}
