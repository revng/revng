/// \file Main.cpp

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <system_error>

#include "revng/Pipeline/ContainerSet.h"
#include "revng/Pipes/ModelGlobal.h"
#include "revng/Pipes/PipelineManager.h"
#include "revng/Pipes/ToolCLOptions.h"
#include "revng/Support/CommandLine.h"
#include "revng/Support/InitRevng.h"

using std::string;
using namespace llvm;
using namespace llvm::cl;
using namespace pipeline;
using namespace ::revng::pipes;
using namespace revng;

static cl::opt<std::string> DiffPath(cl::Positional,
                                     cl::cat(MainCategory),
                                     cl::desc("<model diff>"),
                                     cl::init("-"),
                                     cl::value_desc("model"));

static ToolCLOptions BaseOptions(MainCategory);

static ExitOnError AbortOnError;

int main(int argc, const char *argv[]) {
  using BinaryRef = TupleTreeGlobal<model::Binary>;
  using DiffRef = TupleTreeDiff<model::Binary>;

  revng::InitRevng X(argc, argv);

  HideUnrelatedOptions(MainCategory);
  ParseCommandLineOptions(argc, argv);

  Registry::runAllInitializationRoutines();

  auto Manager = AbortOnError(BaseOptions.makeManager());
  const auto &Ctx = Manager.context();
  auto OriginalModel = *AbortOnError(Ctx.getGlobal<BinaryRef>(ModelGlobalName));

  const auto &Name = ModelGlobalName;
  auto AfterModel = AbortOnError(Ctx.getGlobal<BinaryRef>(Name));

  auto Diff = AbortOnError(deserializeFileOrSTDIN<DiffRef>(DiffPath));
  cantFail(Diff.apply(AfterModel->get()));

  TargetsList OverestimatedTargets;
  for (const Kind &Kind : Manager.getRunner().getKindsRegistry())
    Kind.getInvalidations(Manager.context(),
                          OverestimatedTargets,
                          Diff,
                          OriginalModel,
                          *AfterModel);

  OverestimatedTargets.dump(llvm::outs());

  return EXIT_SUCCESS;
}
