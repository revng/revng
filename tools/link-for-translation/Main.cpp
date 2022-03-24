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
#include "revng/TupleTree/TupleTree.h"

using std::string;
using namespace llvm;
using namespace llvm::cl;

cl::OptionCategory MergeDynamicCategory("revng-merge-dynamic options", "");

static opt<string> Input(Positional,
                         Required,
                         desc("<Input-Binary-Path>"),
                         cat(MergeDynamicCategory));
static opt<string> ModelFile(Positional,
                             Required,
                             desc("<Model-Path>"),
                             cat(MergeDynamicCategory));
static opt<string> ObjectFile(Positional,
                              Required,
                              desc("<Object-File-Path>"),
                              cat(MergeDynamicCategory));
static opt<string> Output("o",
                          init("-"),
                          desc("Output tranlated file"),
                          cat(MergeDynamicCategory));

static opt<bool> Verbose("verbose",
                         desc("Print explanation while running"),
                         cat(MergeDynamicCategory),
                         init(false));

static alias VerboseAlias1("v",
                           desc("Alias for --verbose"),
                           aliasopt(Verbose),
                           cat(MergeDynamicCategory));

static ExitOnError AbortOnError;

int main(int argc, const char *argv[]) {
  HideUnrelatedOptions(MergeDynamicCategory);
  ParseCommandLineOptions(argc, argv);

  auto MaybeModel = TupleTree<model::Binary>::fromFile(ModelFile);
  auto ExpectedModel = llvm::errorOrToExpected(std::move(MaybeModel));
  auto Model = AbortOnError(std::move(ExpectedModel));
  if (Verbose) {
    printLinkForTranslationCommands(llvm::outs(),
                                    *Model,
                                    Input,
                                    ObjectFile,
                                    Output);
    return EXIT_SUCCESS;
  }
  linkForTranslation(*Model, Input, ObjectFile, Output);
  return EXIT_SUCCESS;
}
