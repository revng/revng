/// \file Main.cpp

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <cstdlib>

#include "llvm/ADT/StringRef.h"
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
#include "revng/Pipes/ToolCLOptions.h"
#include "revng/Support/InitRevng.h"

using std::string;
using namespace llvm;
using namespace llvm::cl;
using namespace pipeline;
using namespace ::revng::pipes;

cl::OptionCategory PipelineCategory("revng-invalidate options", "");

static cl::list<string> Targets(Positional,
                                Required,
                                desc("<Targets to invalidate>..."),
                                cat(PipelineCategory));

static opt<bool> DumpPredictedRemovals("dump-invalidations",
                                       desc("dump predicted invalidate "
                                            "targets"),
                                       cat(PipelineCategory));

static opt<bool> DumpFinalStatus("dump-status",
                                 desc("dump status after invalidation "
                                      "targets"),
                                 cat(PipelineCategory));

static ToolCLOptions BaseOptions(PipelineCategory);

static ExitOnError AbortOnError;

static InvalidationMap getInvalidationMap(Runner &Pipeline) {
  InvalidationMap Invalidations;

  const auto &Registry = Pipeline.getKindsRegistry();
  for (llvm::StringRef Target : Targets) {
    auto [StepName, Rest] = Target.split("/");
    auto &ToInvalidate = Invalidations[StepName];
    AbortOnError(parseTarget(ToInvalidate, Rest, Registry));
  }

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
  revng::InitRevng X(argc, argv);

  HideUnrelatedOptions(PipelineCategory);
  ParseCommandLineOptions(argc, argv);

  Registry::runAllInitializationRoutines();

  auto Manager = AbortOnError(BaseOptions.makeManager());

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
