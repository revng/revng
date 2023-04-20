//
// Copyright (c) rev.ng Labs Srl. See LICENSE.md for details.
//

#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Signals.h"

#include "revng/Model/IRHelpers.h"
#include "revng/Model/LoadModelPass.h"
#include "revng/Support/Debug.h"
#include "revng/Support/InitRevng.h"

#include "revng-c/Backend/DecompileFunction.h"
#include "revng-c/Pipes/Kinds.h"
#include "revng-c/Support/IRHelpers.h"

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

static opt<std::string> ModelOverride("m",
                                      Optional,
                                      desc("Model YAML file. It can be used to "
                                           "provide the Model if the input "
                                           "LLVM IR file does not contain it, "
                                           "or to ovverride the Model in the "
                                           "LLVM IR input file."),
                                      cat(DecompileCategory));

static opt<std::string> TargetFunction("t",
                                       Optional,
                                       desc("MetaAddress of the target "
                                            "function to decompile."),
                                       cat(DecompileCategory));

int main(int Argc, const char *Argv[]) {
  revng::InitRevng X(Argc, Argv, Overview, { &DecompileCategory });

  // Get the IR
  llvm::SMDiagnostic Diag;
  LLVMContext Ctx;
  std::unique_ptr<Module> Module = llvm::parseIRFile(InFile, Diag, Ctx);
  if (not Module) {
    Diag.print(Argv[0], llvm::errs());
    std::exit(EXIT_FAILURE);
  }

  // Load the model, either from the ModelOverride file or from the Module
  // metadata.
  TupleTree<model::Binary> Model;
  if (ModelOverride.getNumOccurrences()) {

    auto Buffer = llvm::MemoryBuffer::getFile(ModelOverride);
    if (std::error_code EC = Buffer.getError())
      revng_abort(EC.message().c_str());

    // Deserialize the model override from YAML
    auto M = TupleTree<model::Binary>::deserialize(Buffer.get()->getBuffer());
    if (std::error_code EC = M.getError())
      revng_abort(EC.message().c_str());

    Model = std::move(M.get());
  } else {
    Model = loadModel(*Module);
  }

  if (not Model.verify())
    revng_abort("Invalid Model");

  std::error_code EC;
  llvm::raw_fd_ostream DecompiledOutFile{ OutFile, EC };
  if (EC)
    revng_abort(EC.message().c_str());

  // If the TargetFunction option was passed, we look at the LLVM IR and strip
  // away the bodies of all the functions that are not the target functions. In
  // this way the decompiler does not emit them.
  if (TargetFunction.getNumOccurrences()) {
    MetaAddress Target = MetaAddress::fromString(TargetFunction);

    if (Target == MetaAddress::invalid())
      revng_abort("MetaAddress of the function is invalid");

    for (llvm::Function &F : FunctionTags::Isolated.functions(Module.get()))
      if (Target != getMetaAddressOfIsolatedFunction(F))
        deleteOnlyBody(F);
  }

  using revng::pipes::FunctionStringMap;
  revng::pipes::DecompiledCCodeInYAMLStringMap DecompiledFunctions("" /*Name*/,
                                                                   &Model);

  FunctionMetadataCache Cache;
  decompile(Cache, *Module, *Model, DecompiledFunctions);

  llvm::cantFail(DecompiledFunctions.serialize(DecompiledOutFile));

  return 0;
}
