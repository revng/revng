/// \file Main.cpp

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <string>

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/raw_ostream.h"

#include "revng/Model/Binary.h"
#include "revng/Support/InitRevng.h"
#include "revng/Support/MetaAddress/YAMLTraits.h"
#include "revng/Support/YAMLTraits.h"
#include "revng/TupleTree/TupleTreeDiff.h"

namespace cl = llvm::cl;

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

static cl::opt<std::string> OutputFilename("o",
                                           llvm::cl::init("-"),
                                           llvm::cl::desc("Override output "
                                                          "filename"),
                                           llvm::cl::value_desc("filename"),
                                           llvm::cl::cat(ThisToolCategory));

int main(int Argc, char *Argv[]) {
  revng::InitRevng X(Argc, Argv, "", { &ThisToolCategory });

  llvm::ExitOnError ExitOnError;

  using Type = TupleTree<model::Binary>;
  auto Model = Type::fromFileOrSTDIN(PathModel);
  if (not Model)
    ExitOnError(Model.takeError());

  using TypeDiff = TupleTreeDiff<model::Binary>;
  auto Diff = ExitOnError(fromFileOrSTDIN<TypeDiff>(DiffPath));

  ExitOnError(Diff.apply(*Model));
  ExitOnError(Model->toFile(OutputFilename));

  return EXIT_SUCCESS;
}
