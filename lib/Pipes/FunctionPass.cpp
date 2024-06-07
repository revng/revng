/// \file FunctionPass.cpp
/// Contains the implementation of pipeline passes.

///
//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/Model/LoadModelPass.h"
#include "revng/Pipeline/Context.h"
#include "revng/Pipeline/ExecutionContext.h"
#include "revng/Pipeline/LLVMKind.h"
#include "revng/Pipes/FunctionPass.h"
#include "revng/Pipes/TaggedFunctionKind.h"
#include "revng/Support/FunctionTags.h"

bool pipeline::detail::runOnModule(llvm::Module &Module,
                                   FunctionPassImpl &Pipe) {
  auto &Analysis = Pipe.getAnalysis<pipeline::LoadExecutionContextPass>();

  // Obtain the context
  ExecutionContext *Ctx = Analysis.get();
  revng_assert(Ctx != nullptr);

  // Run the prologue
  auto &ModelWrapper = Pipe.getAnalysis<LoadModelWrapperPass>().get();
  bool Result = Pipe.prologue();

  // Run on individual functions
  using Type = revng::kinds::TaggedFunctionKind;
  auto ToIterOn = Type::getFunctionsAndCommit(*Ctx,
                                              Module,
                                              Analysis.getContainerName());
  for (const auto &[First, Second] : ToIterOn) {
    Result = Pipe.runOnFunction(*First, *Second) or Result;
  }

  Result = Pipe.epilogue() or Result;

  return Result;
}
