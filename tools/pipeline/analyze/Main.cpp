// \file Main.cpp

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <system_error>
#include <utility>

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
#include "revng/Pipeline/LLVMKind.h"
#include "revng/Pipeline/Loader.h"
#include "revng/Pipeline/Step.h"
#include "revng/Pipeline/Target.h"
#include "revng/Pipeline/Target/YAMLTraits.h"
#include "revng/Pipes/ModelGlobal.h"
#include "revng/Pipes/PipelineManager.h"
#include "revng/Pipes/ToolCLOptions.h"
#include "revng/Support/CommandLine.h"
#include "revng/Support/InitRevng.h"
#include "revng/TupleTree/TupleTree.h"

using std::string;
using namespace llvm;
using namespace llvm::cl;
using namespace pipeline;
using namespace ::revng::pipes;
using namespace revng;

static cl::list<string> Arguments(Positional,
                                  ZeroOrMore,
                                  desc("<analysis> <binary>"),
                                  cat(MainCategory));

static OutputPathOpt Output("o",
                            desc("Output filepath of produced model"),
                            cat(MainCategory),
                            init(revng::PathInit::Dash));

static opt<bool> NoApplyModel("no-apply",
                              desc("run the analysis but do not apply it (used "
                                   "to recreate consistent debug "
                                   "configurations)"),
                              cat(MainCategory),
                              init(false));

static opt<std::string> InvalidationsPath("save-invalidations",
                                          desc("path where to save the list of "
                                               "invalidated targets"),
                                          cat(MainCategory),
                                          init(""));

static ToolCLOptions BaseOptions(MainCategory);

static ExitOnError AbortOnError;

static TupleTreeGlobal<model::Binary> &getModel(PipelineManager &Manager) {
  auto &Context = Manager.context();
  const auto &ModelName = revng::ModelGlobalName;
  auto *FinalModel = AbortOnError(Context.getGlobal<ModelGlobal>(ModelName));
  revng_assert(FinalModel != nullptr);
  return *FinalModel;
}

static llvm::Error overrideModel(PipelineManager &Manager,
                                 TupleTree<model::Binary> NewModel) {
  const auto &Name = revng::ModelGlobalName;
  auto *Model(cantFail(Manager.context().getGlobal<revng::ModelGlobal>(Name)));
  Model->get() = std::move(NewModel);
  return llvm::Error::success();
}

int main(int argc, char *argv[]) {
  using revng::FilePath;
  using BinaryRef = TupleTreeGlobal<model::Binary>;

  revng::InitRevng X(argc, argv, "", { &MainCategory });

  Registry::runAllInitializationRoutines();

  auto Manager = AbortOnError(BaseOptions.makeManager());
  const auto &Context = Manager.context();
  auto OriginalModel = *AbortOnError(Context
                                       .getGlobal<BinaryRef>(ModelGlobalName));

  if (Arguments.size() == 0) {
    std::cout << "USAGE: revng-analyze [options] <analysis> <binary>\n\n";
    std::vector<std::pair<std::string, std::string>> Pairs;

    std::cout << "<analysis> can be one of the following analyses list:\n\n";

    for (size_t I = 0; I < Manager.getRunner().getAnalysesListCount(); I++) {
      AnalysesList AL = Manager.getRunner().getAnalysesList(I);
      std::cout << "  " << AL.getName().str() << "\n";
    }

    std::cout << "\n<analysis> can also be one of the following analyses:\n\n";
    for (const auto &Step : Manager.getRunner())
      for (const auto &Analysis : Step.analyses())
        std::cout << "  " << Analysis.getKey().str() << "\n";

    return EXIT_SUCCESS;
  }

  if (Arguments.size() == 1) {
    AbortOnError(createStringError(inconvertibleErrorCode(),
                                   "Expected any number of positional "
                                   "arguments different from 1"));
  }

  auto &InputContainer = Manager.getRunner().begin()->containers()["input"];
  InputPath = Arguments[1];
  AbortOnError(InputContainer.load(FilePath::fromLocalStorage(Arguments[1])));

  auto InvMap = revng::pipes::runAnalysisOrAnalysesList(Manager,
                                                        Arguments[0],
                                                        AbortOnError);

  if (NoApplyModel)
    AbortOnError(overrideModel(Manager, OriginalModel.get()));

  if (not InvalidationsPath.empty()) {
    std::error_code EC;
    ToolOutputFile InvalidationOutput(InvalidationsPath,
                                      EC,
                                      sys::fs::OpenFlags::OF_Text);
    if (EC)
      AbortOnError(llvm::createStringError(EC, EC.message()));

    serialize(InvalidationOutput.os(), InvMap);

    InvalidationOutput.keep();
  }

  AbortOnError(Manager.store());

  auto &FinalModel = getModel(Manager);
  AbortOnError(FinalModel.store(*Output));

  return EXIT_SUCCESS;
}
