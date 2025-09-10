//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/Support/CommandLine.h"

#include "revng/Model/ToolHelpers.h"
#include "revng/Model/TypeSystemPrinter.h"
#include "revng/Support/InitRevng.h"

namespace cl = llvm::cl;

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

  llvm::ExitOnError ExitOnError;

  auto MaybeModel = TupleTree<model::Binary>::fromFileOrSTDIN(InputModulePath);
  if (not MaybeModel)
    ExitOnError(MaybeModel.takeError());

  std::error_code EC;
  llvm::raw_fd_ostream Out(OutputFilename, EC);
  if (EC)
    revng_abort(EC.message().c_str());

  TypeSystemPrinter TSPrinter(Out, **MaybeModel);
  TSPrinter.print();

  return EXIT_SUCCESS;
}
