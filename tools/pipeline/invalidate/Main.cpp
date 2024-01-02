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

static cl::list<string> Targets(Positional,
                                Required,
                                desc("<Targets to invalidate>..."),
                                cat(MainCategory));

static opt<bool> DumpPredictedRemovals("dump-invalidations",
                                       desc("dump predicted invalidate "
                                            "targets"),
                                       cat(MainCategory));

static opt<bool> DumpFinalStatus("dump-status",
                                 desc("dump status after invalidation "
                                      "targets"),
                                 cat(MainCategory));

static ToolCLOptions BaseOptions(MainCategory);

static ExitOnError AbortOnError;

static TargetInStepSet getTargetInStepSet(Runner &Pipeline) {
  TargetInStepSet Invalidations;

  const auto &Registry = Pipeline.getKindsRegistry();
  for (llvm::StringRef Target : Targets) {
    auto [StepName, Rest] = Target.split("/");
    auto &ToInvalidate = Invalidations[StepName];
    auto &Ctx = Pipeline.getContext();
    AbortOnError(parseTarget(Ctx, ToInvalidate, Rest, Registry));
  }

  return Invalidations;
}

static void dumpTargetInStepSet(llvm::raw_ostream &OS,
                                const TargetInStepSet &Map) {
  for (const auto &Pair : Map) {
    OS << Pair.first();
    Pair.second.dump(OS, 1);
  }
}

int main(int argc, char *argv[]) {
  revng::InitRevng X(argc, argv, "", { &MainCategory });

  Registry::runAllInitializationRoutines();

  auto Manager = AbortOnError(BaseOptions.makeManager());

  auto Map = getTargetInStepSet(Manager.getRunner());
  AbortOnError(Manager.getRunner().getInvalidations(Map));

  if (DumpPredictedRemovals) {
    dumpTargetInStepSet(llvm::outs(), Map);
    return EXIT_SUCCESS;
  }

  AbortOnError(Manager.getRunner().invalidate(Map));

  if (DumpFinalStatus)
    Manager.dump();

  AbortOnError(Manager.store());
  return EXIT_SUCCESS;
}
