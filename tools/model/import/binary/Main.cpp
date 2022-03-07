/// \file Main.cpp
/// \brief

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <string>

#include "llvm/Support/CommandLine.h"

#include "revng/Model/Importer/Binary/BinaryImporter.h"
#include "revng/Model/ToolHelpers.h"

using namespace llvm;
using namespace cl;

static OptionCategory ThisToolCategory("Tool options", "");

static opt<std::string> InputFilename(Positional,
                                      cat(ThisToolCategory),
                                      desc("<input file>"),
                                      init("-"),
                                      value_desc("filename"));

static opt<std::string> OutputFilename("o",
                                       cat(ThisToolCategory),
                                       desc("Override output "
                                            "filename"),
                                       init("-"),
                                       value_desc("filename"));

#define DESCRIPTION desc("base address where dynamic objects should be loaded")
static opt<unsigned long long> BaseAddress("base",
                                           DESCRIPTION,
                                           value_desc("address"),
                                           cat(MainCategory),
                                           init(0x400000));
#undef DESCRIPTION

int main(int Argc, char *Argv[]) {
  HideUnrelatedOptions({ &ThisToolCategory });
  ParseCommandLineOptions(Argc, Argv);

  revng_check(BaseAddress % 4096 == 0, "Base address is not page aligned");

  // Open output
  ExitOnError ExitOnError;
  std::error_code EC;
  ToolOutputFile OutputFile(OutputFilename, EC, sys::fs::OpenFlags::OF_Text);
  if (EC)
    ExitOnError(createStringError(EC, EC.message()));

  // Import
  TupleTree<model::Binary> Model;

  ExitOnError(importBinary(Model, InputFilename, BaseAddress));

  // Serialize
  Model.serialize(OutputFile.os());

  Model->verify(true);

  OutputFile.keep();

  return EXIT_SUCCESS;
}