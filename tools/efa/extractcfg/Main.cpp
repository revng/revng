/// \file Main.cpp

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <string>

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/MemoryBuffer.h"

#include "revng/EarlyFunctionAnalysis/IRHelpers.h"
#include "revng/Model/ToolHelpers.h"
#include "revng/Support/IRHelpers.h"

#include "./DecoratedFunction.h"

using namespace llvm;

static cl::opt<std::string> InputModule(cl::Positional,
                                        cl::cat(MainCategory),
                                        cl::desc("<input module>"),
                                        cl::init("-"),
                                        cl::value_desc("filename"));

static cl::opt<std::string> OutputFilename("o",
                                           cl::cat(MainCategory),
                                           cl::init("-"),
                                           llvm::cl::desc("<output file>"),
                                           cl::value_desc("filename"));

int main(int argc, const char **argv) {
  cl::HideUnrelatedOptions({ &MainCategory });
  cl::ParseCommandLineOptions(argc, argv);

  auto BufOrError = MemoryBuffer::getFileOrSTDIN(InputModule);
  if (std::error_code EC = BufOrError.getError())
    ExitOnError("Unable to read LLVM IR module: " + EC.message() + "\n");

  SMDiagnostic Diagnostic;
  LLVMContext Context;
  auto Module = llvm::parseIR(*BufOrError.get(), Diagnostic, Context);

  TupleTree<model::Binary> Model;
  if (hasModel(*Module))
    Model = loadModel(*Module);
  else
    ExitOnError("Unable to extract model\n");

  SortedVector<revng::DecoratedFunction> DecoratedFunctions;
  for (BasicBlock &BB : *Module->getFunction("root")) {
    llvm::Instruction *Term = BB.getTerminator();
    auto *FMMDNode = Term->getMetadata(FunctionMetadataMDName);
    if (not FMMDNode)
      continue;

    efa::FunctionMetadata FM = *extractFunctionMetadata(&BB).get();
    auto &Function = Model->Functions.at(getBasicBlockPC(&BB));
    DecoratedFunctions.insert({ Function.OriginalName, Function.Type, FM });
  }

  ExitOnError AbortOnError;
  AbortOnError((serializeToFile(DecoratedFunctions, OutputFilename)));

  return EXIT_SUCCESS;
}
