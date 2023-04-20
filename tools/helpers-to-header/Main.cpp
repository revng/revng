//
// Copyright (c) rev.ng Labs Srl. See LICENSE.md for details.
//

#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Signals.h"
#include "llvm/Support/SourceMgr.h"

#include "revng/Support/Assert.h"
#include "revng/Support/InitRevng.h"

#include "revng-c/HeadersGeneration/HelpersToHeader.h"

using namespace llvm::cl;

static OptionCategory HelpersToHeaderCategory("revng-helpers-to-headers "
                                              "options");

static const char *Overview = "Standalone tool that ingests an LLVM module and "
                              "produces a C11 header file\n"
                              "with all the declarations of types and "
                              "functions associated to QEMU and revng\n"
                              "helpers.";

static opt<std::string> OutFile("o",
                                init("-" /* for stdout */),
                                desc("Output C header"),
                                cat(HelpersToHeaderCategory));

static opt<std::string> InFile("i",
                               init("-" /* for stdin */),
                               desc("Input LLVM Module"),
                               cat(HelpersToHeaderCategory));

using llvm::LLVMContext;
using llvm::Module;

int main(int Argc, const char *Argv[]) {
  revng::InitRevng X(Argc, Argv, Overview, { &HelpersToHeaderCategory });

  auto Buffer = llvm::MemoryBuffer::getFileOrSTDIN(InFile);
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
  llvm::raw_fd_ostream Header{ OutFile, EC };
  if (EC)
    revng_abort(EC.message().c_str());

  dumpHelpersToHeader(*M, Header);

  Header.flush();
  EC = Header.error();
  if (EC)
    revng_abort(EC.message().c_str());

  return 0;
}
