/// \file Main.cpp

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <cstdlib>

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/DynamicLibrary.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_os_ostream.h"

#include "revng/Recompile/LinkForTranslation.h"
#include "revng/Support/InitRevng.h"
#include "revng/TupleTree/TupleTree.h"

using std::string;
using namespace llvm;
using namespace llvm::cl;

class StringPositionalArgument : public opt<string> {
public:
  StringPositionalArgument(const char *Description) :
    opt<string>(Positional, Required, desc(Description), cat(MainCategory)) {}
};

StringPositionalArgument Input("Input binary");
StringPositionalArgument ModelFile("Input model");
StringPositionalArgument ObjectFile("Object file");

auto OutputDescription = desc("Output translated file");
static opt<string> Output("o", init("-"), OutputDescription, cat(MainCategory));

static opt<bool> Verbose("verbose",
                         desc("Print explanation while running"),
                         cat(MainCategory),
                         init(false));

static opt<bool> DryRun("dry-run",
                        desc("Don't do anything. Useful with --verbose"),
                        cat(MainCategory),
                        init(false));

static ExitOnError AbortOnError;

int main(int argc, char *argv[]) {
  revng::InitRevng X(argc, argv, "", { &MainCategory });

  auto MaybeModel = TupleTree<model::Binary>::fromFile(ModelFile);
  auto ExpectedModel = llvm::errorOrToExpected(std::move(MaybeModel));
  auto Model = AbortOnError(std::move(ExpectedModel));

  if (Verbose) {
    printLinkForTranslationCommands(llvm::outs(),
                                    *Model,
                                    Input,
                                    ObjectFile,
                                    Output);
  }

  if (not DryRun)
    linkForTranslation(*Model, Input, ObjectFile, Output);

  return EXIT_SUCCESS;
}
