//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <fstream>
#include <iostream>
#include <string>

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Signals.h"

#include "revng/Model/Binary.h"
#include "revng/Model/DumpSchema.h"

static llvm::cl::OptionCategory DumpSchemaCategory("DumpSchemaOptions");

static const char *Overview = R"(Dump rev.ng model JSON schema)";

static llvm::cl::opt<std::string> OutputPath("output",
                                             llvm::cl::cat(DumpSchemaCategory),
                                             llvm::cl::Optional,
                                             llvm::cl::desc("Output path "
                                                            "(default "
                                                            "stdout)"));

int main(int Argc, const char *Argv[]) {

  // Enable LLVM stack trace
  llvm::sys::PrintStackTraceOnErrorSignal(Argv[0]);

  // Hide options not related to this tool
  llvm::cl::HideUnrelatedOptions({ &DumpSchemaCategory });

  // Parse command line
  bool Ok = llvm::cl::ParseCommandLineOptions(Argc,
                                              Argv,
                                              Overview,
                                              &llvm::errs());
  if (not Ok) {
    std::exit(EXIT_FAILURE);
  }

  SchemaDumper<model::Binary> D;
  if (!OutputPath.empty()) {
    std::ofstream Output{ OutputPath.getValue() };
    if (!Output) {
      std::cerr << "Could not open output file\n";
      std::exit(EXIT_FAILURE);
    }
    D.dumpSchema(Output);
  } else {
    D.dumpSchema(std::cout);
  }

  return 0;
}
