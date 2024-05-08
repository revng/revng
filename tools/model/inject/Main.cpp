/// \file Main.cpp

//
// This file is distributed under the MIT License. See LICENSE.mit for details.
//

#include <string>

#include "llvm/Support/CommandLine.h"

#include "revng/Model/ToolHelpers.h"
#include "revng/Support/InitRevng.h"

using namespace llvm;

static cl::OptionCategory ThisToolCategory("Tool options", "");

static ModelOutputOptions<false> Options(ThisToolCategory);

static cl::opt<std::string> NewModelPath(cl::Positional,
                                         cl::cat(ThisToolCategory),
                                         cl::desc("<new model file>"),
                                         cl::value_desc("newmodel"));

static cl::opt<std::string> InputModulePath(cl::Positional,
                                            cl::cat(ThisToolCategory),
                                            cl::desc("<input module file>"),
                                            cl::init("-"),
                                            cl::value_desc("module"));

int main(int Argc, char *Argv[]) {
  revng::InitRevng X(Argc, Argv, "", { &ThisToolCategory });

  auto OldModel = ModelInModule::loadModule(InputModulePath);
  auto NewModel = ModelInModule::loadYAML(NewModelPath);

  ExitOnError ExitOnError;
  if (not OldModel)
    ExitOnError(OldModel.takeError());
  if (not NewModel)
    ExitOnError(NewModel.takeError());

  OldModel->Model = std::move(NewModel->Model);
  ExitOnError(OldModel->save(Options.getPath(),
                             Options.getDesiredOutput(OldModel->hasModule())));
}
