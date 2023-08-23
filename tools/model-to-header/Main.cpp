//
// Copyright (c) rev.ng Labs Srl. See LICENSE.md for details.
//

#include <memory>
#include <string>

#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Signals.h"
#include "llvm/Support/SourceMgr.h"

#include "revng/Model/LoadModelPass.h"
#include "revng/Model/ToolHelpers.h"
#include "revng/Support/Assert.h"
#include "revng/Support/InitRevng.h"

#include "revng-c/HeadersGeneration/ModelToHeader.h"

using namespace llvm::cl;

static OptionCategory ModelToHeaderCategory("revng-model-to-header options");

static const char *Overview = "Standalone tool that ingests an instance of "
                              "revng's model, embedded into an\n"
                              "llvm IR file, and produces a C11 header file "
                              "with all the declarations of\n"
                              "types and functions associated to the model's "
                              "type system.";

static opt<std::string> OutFile("o",
                                init("-" /* for stdout */),
                                desc("Output C header"),
                                cat(ModelToHeaderCategory));

static opt<std::string> InFile("i",
                               init("-" /* for stdin */),
                               desc("Input file with Model"),
                               cat(ModelToHeaderCategory));

using llvm::LLVMContext;
using llvm::Module;

int main(int Argc, const char *Argv[]) {
  revng::InitRevng X(Argc, Argv, Overview, { &ModelToHeaderCategory });

  auto Buffer = llvm::MemoryBuffer::getFileOrSTDIN(InFile);
  if (std::error_code EC = Buffer.getError())
    revng_abort(EC.message().c_str());

  // Load the module either in yaml or bitcode format
  auto ModelWrapper = llvm::cantFail(ModelInModule::load(**Buffer));
  const model::Binary &Model = ModelWrapper.getReadOnlyModel();
  if (not Model.verify()) {
    llvm::errs() << "Invalid Model\n";
    std::exit(EXIT_FAILURE);
  }

  std::error_code EC;
  llvm::raw_fd_ostream Header{ OutFile, EC };
  if (EC)
    revng_abort(EC.message().c_str());

  dumpModelToHeader(Model, Header, {});

  Header.flush();
  EC = Header.error();
  if (EC)
    revng_abort(EC.message().c_str());

  return 0;
}
