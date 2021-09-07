//
// This file is distributed under the MIT License. See LICENSE.md for details.
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
#include "revng/Support/Assert.h"

static llvm::cl::OptionCategory VerifyModelCategory("VerifyModelOptions");

static const char *Overview = R"(
  Standalone tool to verify revng model in `.ll` or `.yml` format.
  Ingests input from STDIN.
  On sucesss, exit status is 0.
  On failure, it reports errors and exit status is != 0.
)";

static llvm::cl::opt<bool> InputIsYAML("yaml",
                                       llvm::cl::cat(VerifyModelCategory),
                                       llvm::cl::Optional,
                                       llvm::cl::init(false),
                                       llvm::cl::desc("Set to true if input is "
                                                      "yaml, not llvm IR"));

using llvm::LLVMContext;
using llvm::Module;

int main(int Argc, const char *Argv[]) {

  // Enable LLVM stack trace
  llvm::sys::PrintStackTraceOnErrorSignal(Argv[0]);

  // Hide options not related to this tool
  llvm::cl::HideUnrelatedOptions({ &VerifyModelCategory });

  // Parse command line
  bool Ok = llvm::cl::ParseCommandLineOptions(Argc,
                                              Argv,
                                              Overview,
                                              &llvm::errs());
  if (not Ok)
    std::exit(EXIT_FAILURE);

  auto Buffer = llvm::MemoryBuffer::getSTDIN();
  if (std::error_code EC = Buffer.getError())
    revng_abort(EC.message().c_str());

  // Try to load the module from stdin, either in yaml or bitcode format
  TupleTree<model::Binary> Model;
  if (InputIsYAML) {

    // If input is YAML, get it from yaml
    auto M = TupleTree<model::Binary>::deserialize(Buffer.get()->getBuffer());
    if (std::error_code EC = M.getError())
      revng_abort(EC.message().c_str());

    Model = std::move(M.get());
  } else {
    // If we're loading from LLVM IR, do it lazily, since we don't really need
    // to deserialize function bodies for getting the model.
    llvm::SMDiagnostic Diag;
    LLVMContext Ctx;
    std::unique_ptr<Module> M = llvm::getLazyIRModule(std::move(Buffer.get()),
                                                      Diag,
                                                      Ctx);
    if (not M) {
      Diag.print(Argv[0], llvm::errs());
      std::exit(EXIT_FAILURE);
    }

    Model = loadModel(*M);
  }

  revng_check(Model->verify(true));

  return 0;
}

