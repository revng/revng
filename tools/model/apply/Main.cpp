/// \file Main.cpp

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <string>

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/raw_ostream.h"

#include "revng/Model/ToolHelpers.h"
#include "revng/Support/InitRevng.h"
#include "revng/Support/MetaAddress/YAMLTraits.h"
#include "revng/Support/YAMLTraits.h"
#include "revng/TupleTree/TupleTreeDiff.h"

using namespace llvm;

static cl::OptionCategory ThisToolCategory("Tool options", "");

static cl::opt<std::string> PathModel(cl::Positional,
                                      cl::cat(ThisToolCategory),
                                      cl::desc("<input model>"),
                                      cl::value_desc("model"));

static cl::opt<std::string> DiffPath(cl::Positional,
                                     cl::cat(ThisToolCategory),
                                     cl::desc("<model diff>"),
                                     cl::init("-"),
                                     cl::value_desc("model"));

static ModelOutputOptions<false> Options(ThisToolCategory);

int main(int Argc, char *Argv[]) {
  revng::InitRevng X(Argc, Argv, "", { &ThisToolCategory });

  ExitOnError ExitOnError;

  auto Model = ModelInModule::load(PathModel);
  if (not Model)
    ExitOnError(Model.takeError());

  using Type = TupleTreeDiff<model::Binary>;
  auto Diff = ExitOnError(deserializeFileOrSTDIN<Type>(DiffPath));

  ExitOnError(Diff.apply(Model->getWriteableModel()));

  auto DesiredOutput = Options.getDesiredOutput(Model->hasModule());
  ExitOnError(Model->save(Options.getPath(), DesiredOutput));

  return EXIT_SUCCESS;
}
