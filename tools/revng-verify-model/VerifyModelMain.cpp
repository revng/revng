/// \file VerifyModelMain.cpp
/// \brief Implements the main routine for the standalone tool that verifies a
/// given model is valid

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <fstream>
#include <memory>

#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Signals.h"
#include "llvm/Support/SourceMgr.h"

#include "revng/Model/LoadModelPass.h"
#include "revng/Model/Processing.h"
#include "revng/Support/Assert.h"

using namespace llvm;
using namespace llvm::cl;

static OptionCategory VerifyModelCategory("VerifyModelOptions");

static const char *Overview = R"LLVM(
  Standalone tool to verify revng model in YAML format (use revng-dump-model to
  extract the YAML model from an LLVM IR file produced by revng-lift).
  Ingests input from STDIN.
  On success, writes back the validated model to stdout and exits with status 0.
  On failure, it reports errors and exit status is != 0.
)LLVM";

static opt<std::string> InvalidTypeIDs("invalid-types",
                                       cat(VerifyModelCategory),
                                       cl::Optional,
                                       desc("Path to list of typerefs to "
                                            "remove"));

static opt<bool> Sanitize("sanitize",
                          cat(VerifyModelCategory),
                          cl::Optional,
                          init(false),
                          desc("Sanitize model names before validation"));

int main(int Argc, const char *Argv[]) {

  // Enable LLVM stack trace
  sys::PrintStackTraceOnErrorSignal(Argv[0]);

  // Hide options not related to this tool
  HideUnrelatedOptions({ &VerifyModelCategory });

  // Parse command line
  bool Ok = ParseCommandLineOptions(Argc, Argv, Overview, &errs());
  if (not Ok)
    return EXIT_FAILURE;

  auto Buffer = MemoryBuffer::getSTDIN();
  if (std::error_code EC = Buffer.getError())
    revng_abort(EC.message().c_str());

  // Try to load the module from stdin
  auto M = TupleTree<model::Binary>::deserialize(Buffer.get()->getBuffer());
  if (std::error_code EC = M.getError()) {
    dbg << "Could not deserialize the given model\n";
    dbg << EC.message().c_str();
    return EXIT_FAILURE;
  }

  auto Model = std::move(M.get());

  if (not InvalidTypeIDs.empty()) {
    std::set<const model::Type *> InvalidTypes;
    std::ifstream InputFile(InvalidTypeIDs);

    if (!InputFile) {
      dbg << "Could not open invalid types file " << InvalidTypeIDs << "\n";
      return EXIT_FAILURE;
    }

    for (std::string line; getline(InputFile, line);) {
      auto Type = model::TypePath::fromString(Model.get(), line).get();
      InvalidTypes.insert(Type);
    }

    unsigned int DeletedTypes = model::dropTypesDependingOnTypes(Model,
                                                                 InvalidTypes);
    dbg << "Deleted " << DeletedTypes << " types out of " << Model->Types.size()
        << "\n";
  }

  if (Sanitize) {
    model::sanitizeCustomNames(Model);
    model::deduplicateNames(Model);
  }

  if (not Model->verify(true))
    return EXIT_FAILURE;

  serialize(outs(), *Model);

  return EXIT_SUCCESS;
}
