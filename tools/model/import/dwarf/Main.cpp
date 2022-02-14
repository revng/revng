/// \file Main.cpp
/// \brief

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <string>

#include "llvm/Support/CommandLine.h"

#include "revng/Model/Importer/Dwarf/DwarfImporter.h"
#include "revng/Model/ToolHelpers.h"

using namespace llvm;

static cl::OptionCategory ThisToolCategory("Tool options", "");

static cl::opt<std::string> InputFilename(cl::Positional,
                                          cl::cat(ThisToolCategory),
                                          cl::desc("<input file>"),
                                          cl::init("-"),
                                          cl::value_desc("filename"));

static cl::opt<std::string> OutputFilename("o",
                                           cl::cat(ThisToolCategory),
                                           llvm::cl::desc("Override output "
                                                          "filename"),
                                           llvm::cl::init("-"),
                                           llvm::cl::value_desc("filename"));

int main(int Argc, char *Argv[]) {
  cl::HideUnrelatedOptions({ &ThisToolCategory });
  cl::ParseCommandLineOptions(Argc, Argv);

  // Open output
  ExitOnError ExitOnError;
  std::error_code EC;
  llvm::ToolOutputFile OutputFile(OutputFilename,
                                  EC,
                                  sys::fs::OpenFlags::OF_Text);
  if (EC)
    ExitOnError(llvm::createStringError(EC, EC.message()));

  // Import
  TupleTree<model::Binary> Model;
  DwarfImporter Importer(Model);
  Importer.import(InputFilename);

  // Serialize
  Model.serialize(OutputFile.os());

  OutputFile.keep();

  return EXIT_SUCCESS;
}
