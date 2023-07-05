/// \file Main.cpp

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <string>

#include "llvm/Support/CommandLine.h"

#include "revng/Model/Importer/Binary/BinaryImporter.h"
#include "revng/Model/Importer/Binary/Options.h"
#include "revng/Model/Importer/DebugInfo/DwarfImporter.h"
#include "revng/Model/ToolHelpers.h"
#include "revng/Support/CommandLine.h"
#include "revng/Support/InitRevng.h"

using namespace llvm;
using namespace cl;

static opt<std::string> InputFilename(Positional,
                                      cat(MainCategory),
                                      desc("<input file>"),
                                      init("-"),
                                      value_desc("filename"));

static opt<std::string> OutputFilename("o",
                                       cat(MainCategory),
                                       desc("Override output "
                                            "filename"),
                                       init("-"),
                                       value_desc("filename"));

int main(int Argc, char *Argv[]) {
  revng::InitRevng X(Argc, Argv, "", { &MainCategory });

  // Open output
  ExitOnError ExitOnError;
  std::error_code EC;
  ToolOutputFile OutputFile(OutputFilename, EC, sys::fs::OpenFlags::OF_Text);
  if (EC)
    ExitOnError(createStringError(EC, EC.message()));

  // Import
  TupleTree<model::Binary> Model;

  const ImporterOptions &Options = importerOptions();
  ExitOnError(importBinary(Model, InputFilename, Options));

  revng_check(Options.BaseAddress % 4096 == 0,
              "Base address is not page aligned");

  if (!Options.AdditionalDebugInfoPaths.empty()) {
    DwarfImporter Importer(Model);
    for (const std::string &Path : Options.AdditionalDebugInfoPaths)
      Importer.import(Path, Options);
  }

  // Serialize
  Model.serialize(OutputFile.os());

  Model->verify(true);

  OutputFile.keep();

  return EXIT_SUCCESS;
}
