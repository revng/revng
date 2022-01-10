//
// Copyright (c) rev.ng Srls. See LICENSE.md for details.
//

#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Signals.h"
#include "llvm/Support/SourceMgr.h"

#include "revng/Support/Assert.h"

#include "revng-c/HeadersGeneration/HelpersToHeader.h"

static llvm::cl::OptionCategory HelpersToHeaderCategory("HelpersToHeaderOption"
                                                        "s");

static const char *Overview = "Standalone tool that ingests an LLVM module "
                              "and produces a C11 header file with all the "
                              "declarations of\ntypes and functions associated "
                              "to QEMU and revng helpers.";

using llvm::LLVMContext;
using llvm::Module;

int main(int Argc, const char *Argv[]) {

  // Enable LLVM stack trace
  llvm::sys::PrintStackTraceOnErrorSignal(Argv[0]);

  // Hide options not related to this tool
  llvm::cl::HideUnrelatedOptions({ &HelpersToHeaderCategory });

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

  // Load IR module
  llvm::SMDiagnostic Diag;
  LLVMContext Ctx;
  std::unique_ptr<Module> M = llvm::parseIR(*Buffer.get(), Diag, Ctx);
  if (not M) {
    Diag.print(Argv[0], llvm::errs());
    std::exit(EXIT_FAILURE);
  }

  std::error_code EC;
  llvm::raw_fd_ostream Header{ "-" /* for stdout */, EC };
  if (EC)
    revng_abort(EC.message().c_str());

  dumpHelpersToHeader(*M, Header);

  return 0;
}
