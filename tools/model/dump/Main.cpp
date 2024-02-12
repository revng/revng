/// \file Main.cpp

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <string>

#include "llvm/Support/CommandLine.h"

#include "revng/Model/ToolHelpers.h"
#include "revng/Support/InitRevng.h"

using namespace llvm;

static cl::OptionCategory ThisToolCategory("Tool options", "");

cl::opt<std::string> OutputFilename("o",
                                    cl::cat(ThisToolCategory),
                                    cl::desc("Override output filename"),
                                    cl::init("-"),
                                    cl::value_desc("filename"));

static cl::opt<std::string> InputModulePath(cl::Positional,
                                            cl::cat(ThisToolCategory),
                                            cl::desc("<input module file>"),
                                            cl::init("-"),
                                            cl::value_desc("module"));

int main(int Argc, char *Argv[]) {
  revng::InitRevng X(Argc, Argv, "", { &ThisToolCategory });

  ExitOnError ExitOnError;

  using Model = TupleTree<model::Binary>;
  auto MaybeModel = errorOrToExpected(Model::fromFileOrSTDIN(InputModulePath));
  if (not MaybeModel)
    ExitOnError(MaybeModel.takeError());

  ExitOnError(MaybeModel->toFile(OutputFilename));

  return EXIT_SUCCESS;
}
