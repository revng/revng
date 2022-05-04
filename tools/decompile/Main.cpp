//
// Copyright (c) rev.ng Labs Srl. See LICENSE.md for details.
//

#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Signals.h"

#include "revng/Model/LoadModelPass.h"
#include "revng/Support/Debug.h"

#include "revng-c/Backend/DecompileFunction.h"
#include "revng-c/Pipes/Kinds.h"

using namespace llvm::cl;

using llvm::LLVMContext;
using llvm::Module;

static OptionCategory DecompileCategory("revng-decompile options");

static const char *Overview = "Ingests an LLVM IR file, with Model metadata, "
                              "and emits a YAML file, containing a mapping "
                              "from Functions' MetaAddresses to the decompiled "
                              "C code generated for each Function.";

static opt<std::string> OutFile("o",
                                Optional,
                                init("-" /* for stdout */),
                                desc("Output YAML file. It contains a mapping "
                                     "from a Function's MetaAddress to the "
                                     "decompiled C code generated for that "
                                     "Function."),
                                cat(DecompileCategory));

static opt<std::string> InFile("i",
                               Optional,
                               init("-" /* for stdin */),
                               desc("Input LLVM IR file. The file needs to "
                                    "contain valid Model metadata."),
                               cat(DecompileCategory));

int main(int Argc, const char *Argv[]) {

  // Enable LLVM stack trace
  llvm::sys::PrintStackTraceOnErrorSignal(Argv[0]);

  // Hide options not related to this tool
  llvm::cl::HideUnrelatedOptions({ &DecompileCategory });

  // Parse command line
  bool Ok = llvm::cl::ParseCommandLineOptions(Argc,
                                              Argv,
                                              Overview,
                                              &llvm::errs());
  if (not Ok)
    std::exit(EXIT_FAILURE);

  auto Buffer = llvm::MemoryBuffer::getFileOrSTDIN(InFile);
  if (std::error_code EC = Buffer.getError())
    revng_abort(EC.message().c_str());

  // Get Model from the IR
  llvm::SMDiagnostic Diag;
  LLVMContext Ctx;
  std::unique_ptr<Module> M = llvm::getLazyIRModule(std::move(Buffer.get()),
                                                    Diag,
                                                    Ctx);
  if (not M) {
    Diag.print(Argv[0], llvm::errs());
    std::exit(EXIT_FAILURE);
  }

  TupleTree<model::Binary> Model = loadModel(*M);

  if (not Model.verify()) {
    llvm::errs() << "Invalid Model\n";
    std::exit(EXIT_FAILURE);
  }

  std::error_code EC;
  llvm::raw_fd_ostream DecompiledOutFile{ OutFile, EC };
  if (EC)
    revng_abort(EC.message().c_str());

  using revng::pipes::FunctionStringMap;
  FunctionStringMap DecompiledFunctions("" /*Name*/,
                                        "application/"
                                        "x.yaml.c.decompiled",
                                        revng::pipes::DecompiledToC,
                                        *Model);

  decompile(*M, *Model, DecompiledFunctions);

  llvm::cantFail(DecompiledFunctions.serialize(DecompiledOutFile));

  return 0;
}
