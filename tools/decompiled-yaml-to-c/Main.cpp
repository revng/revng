//
// Copyright (c) rev.ng Labs Srl. See LICENSE.md for details.
//

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Signals.h"

#include "revng/Support/Debug.h"
#include "revng/Support/InitRevng.h"

#include "revng-c/Backend/DecompiledYAMLToC.h"
#include "revng-c/Pipes/Kinds.h"

using namespace llvm::cl;

using llvm::LLVMContext;
using llvm::Module;

static OptionCategory YAMLToCCategory("revng-decompiled-yaml-to-c options");

static const char *Overview = "Ingests an YAML filed containing a mapping from "
                              "Functions' MetaAddresses to the associated "
                              "decompiled C code, and generates a valid file C "
                              "containing all the Functions' bodies.";

static opt<std::string> OutFile("o",
                                Optional,
                                init("-" /* for stdout */),
                                desc("Output YAML file. It contains a mapping "
                                     "from a Function's MetaAddress to the "
                                     "decompiled C code generated for that "
                                     "Function."),
                                cat(YAMLToCCategory));

static opt<std::string> InFile("i",
                               Optional,
                               init("-" /* for stdin */),
                               desc("A YAML file containing a map. The key of "
                                    "the map are Functions' MetaAddresses, and "
                                    "the value is an escaped string containing "
                                    "the C code of each decompiled function."),
                               cat(YAMLToCCategory));

static opt<std::string> TargetFunction("t",
                                       Optional,
                                       desc("MetaAddress of the target "
                                            "function to decompile."),
                                       cat(YAMLToCCategory));

int main(int Argc, const char *Argv[]) {
  revng::InitRevng X(Argc, Argv, Overview, { &YAMLToCCategory });

  // Open the output file.
  std::error_code OutEC;
  llvm::raw_fd_ostream OutputCCodeFile{ OutFile, OutEC };
  if (OutEC)
    revng_abort(OutEC.message().c_str());

  // Read the input file into a buffer
  auto InputBuffer = llvm::MemoryBuffer::getFileOrSTDIN(InFile);
  if (std::error_code InEC = InputBuffer.getError())
    revng_abort(InEC.message().c_str());

  using revng::pipes::FunctionStringMap;
  TupleTree<model::Binary> Model;

  revng::pipes::DecompiledCCodeInYAMLStringMap Functions("" /*Name*/, &Model);

  // Deserialize the map
  llvm::cantFail(Functions.deserialize(**InputBuffer));

  // Setup the target function to decompile. An empty Targets set means all the
  // functions.
  std::set<MetaAddress> Targets;

  if (TargetFunction.getNumOccurrences()) {
    // If the TargetFunction option was passed, only print that function body
    MetaAddress Target = MetaAddress::fromString(TargetFunction);

    if (Target == MetaAddress::invalid())
      revng_abort("MetaAddress of the function is invalid");

    if (not Functions.contains(Target))
      revng_abort("Cannot find target function");

    Targets.insert(Target);
  }

  printSingleCFile(OutputCCodeFile, Functions, Targets);

  OutputCCodeFile.flush();
  OutEC = OutputCCodeFile.error();
  if (OutEC)
    revng_abort(OutEC.message().c_str());

  return 0;
}
