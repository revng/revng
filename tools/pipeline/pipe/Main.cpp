// \file Main.cpp

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <iostream>

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/DynamicLibrary.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/PluginLoader.h"
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
#include "revng/Support/InitRevng.h"

using std::string;
using namespace llvm;
using namespace llvm::cl;
using namespace pipeline;
using namespace ::revng::pipes;
using namespace ::revng;

static cl::list<string> Arguments(Positional,
                                  ZeroOrMore,
                                  desc("<pipe> <containers>..."),
                                  cat(MainCategory));

static llvm::cl::list<std::string>
  InputPipeline("P", llvm::cl::desc("<Pipeline>"), llvm::cl::cat(MainCategory));

static llvm::cl::list<std::string> EnablingFlags("f",
                                                 llvm::cl::desc("list of "
                                                                "pipeline "
                                                                "enabling "
                                                                "flags"),
                                                 llvm::cl::cat(MainCategory));

static llvm::cl::alias A1("l",
                          llvm::cl::desc("Alias for --load"),
                          llvm::cl::aliasopt(llvm::LoadOpt),
                          llvm::cl::cat(MainCategory));

static ExitOnError AbortOnError;

int main(int argc, char *argv[]) {
  using revng::FilePath;

  revng::InitRevng X(argc, argv, "", { &MainCategory });

  Registry::runAllInitializationRoutines();

  auto
    Manager = AbortOnError(revng::pipes::PipelineManager::create(InputPipeline,
                                                                 EnablingFlags,
                                                                 ""));
  const pipeline::Loader &Loader = Manager.getLoader();
  const llvm::StringMap<PipeWrapper> &PipesMap = Loader.getPipes();

  if (Arguments.size() <= 1 or PipesMap.count(Arguments[0]) == 0) {
    llvm::outs() << "USAGE: revng-pipe [options] <pipe> <model> <args>...\n\n";
    llvm::outs() << "<pipe> can be one of:\n\n";

    for (auto &Pair : PipesMap) {
      llvm::outs() << "revng pipe " << Pair.first() << " <model> ";

      for (size_t I = 0; I < Pair.second.Pipe->getContainerArgumentsCount();
           I++) {
        llvm::outs() << "<" << Pair.second.Pipe->getContainerName(I) << "> ";
      }
      llvm::outs() << "\n";
    }

    return EXIT_SUCCESS;
  }

  const auto &Name = ModelGlobalName;
  auto *Model(cantFail(Manager.context().getGlobal<ModelGlobal>(Name)));
  AbortOnError(Model->load(FilePath::fromLocalStorage(Arguments[1])));

  const auto &Pipe = PipesMap.find(Arguments[0])->second.Pipe;
  std::vector<std::string> DefaultNames;
  pipeline::ContainerSet Set;

  if (Arguments.size() - 2 != Pipe->getContainerArgumentsCount()) {
    llvm::errs() << "Pipe " << Arguments[0] << " required "
                 << Pipe->getContainerArgumentsCount() << " arguments\n";
    return EXIT_FAILURE;
  }
  for (size_t I = 0; I < Pipe->getContainerArgumentsCount(); I++) {
    DefaultNames.push_back((llvm::Twine("arg") + llvm::Twine(I)).str());
  }

  PipeWrapper ClonedPipe(Pipe, DefaultNames);
  for (size_t I = 0; I < Pipe->getContainerArgumentsCount(); I++) {
    llvm::StringRef Name = ClonedPipe.Pipe->getContainerName(I);
    auto &Factory = Loader.getContainerFactory(Name);
    Set.add(DefaultNames[I], Factory, Factory(DefaultNames[I]));
    AbortOnError(Set.at(DefaultNames[I])
                   .load(FilePath::fromLocalStorage(Arguments[I + 2])));
  }
  auto Enumeration = Set.enumerate();
  ExecutionContext ExecutionContext(Manager.context(),
                                    &ClonedPipe,
                                    Enumeration);

  AbortOnError(ClonedPipe.Pipe->run(ExecutionContext, Set));

  for (size_t I = 0; I < Pipe->getContainerArgumentsCount(); I++) {
    if (not Pipe->isContainerArgumentConst(I))
      AbortOnError(Set.at(DefaultNames[I])
                     .store(FilePath::fromLocalStorage(Arguments[I + 2])));
  }

  return EXIT_SUCCESS;
}
