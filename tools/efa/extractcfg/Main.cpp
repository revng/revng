/// \file Main.cpp

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <string>

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/MemoryBuffer.h"

#include "revng/EarlyFunctionAnalysis/FunctionMetadataCache.h"
#include "revng/Model/ToolHelpers.h"
#include "revng/Support/IRHelpers.h"
#include "revng/Support/InitRevng.h"

#include "DecoratedFunction.h"

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

int main(int argc, char *argv[]) {
  revng::InitRevng X(argc, argv, "", { &MainCategory });

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
  auto *RootFunction = Module->getFunction("root");
  revng_assert(RootFunction != nullptr);

  FunctionMetadataCache Cache;
  if (not RootFunction->isDeclaration()) {
    for (BasicBlock &BB : *Module->getFunction("root")) {
      llvm::Instruction *Term = BB.getTerminator();
      auto *FMMDNode = Term->getMetadata(FunctionMetadataMDName);
      if (not FMMDNode)
        continue;

      const efa::FunctionMetadata &FM = Cache.getFunctionMetadata(&BB);
      auto &Function = Model->Functions().at(FM.Entry());
      MutableSet<model::FunctionAttribute::Values> Attributes;
      for (auto &Entry : Function.Attributes())
        Attributes.insert(Entry);
      revng::DecoratedFunction NewFunction(FM.Entry(),
                                           Function.OriginalName(),
                                           FM,
                                           Attributes);
      DecoratedFunctions.insert(std::move(NewFunction));
    }
  }

  for (Function &F : FunctionTags::Isolated.functions(Module.get())) {
    auto *FMMDNode = F.getMetadata(FunctionMetadataMDName);
    const efa::FunctionMetadata &FM = Cache.getFunctionMetadata(&F);
    if (not FMMDNode or DecoratedFunctions.contains(FM.Entry()))
      continue;

    auto &Function = Model->Functions().at(FM.Entry());
    MutableSet<model::FunctionAttribute::Values> Attributes;
    for (auto &Entry : Function.Attributes())
      Attributes.insert(Entry);
    revng::DecoratedFunction NewFunction(FM.Entry(),
                                         Function.OriginalName(),
                                         FM,
                                         Attributes);
    DecoratedFunctions.insert(std::move(NewFunction));
  }

  ExitOnError AbortOnError;
  AbortOnError((serializeToFile(DecoratedFunctions, OutputFilename)));

  return EXIT_SUCCESS;
}
