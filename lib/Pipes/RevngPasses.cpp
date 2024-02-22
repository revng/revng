/// \file RevngPasses.cpp
/// Contains the implementation of pipeline passes.

///
//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/Model/LoadModelPass.h"
#include "revng/Pipeline/Context.h"
#include "revng/Pipeline/ExecutionContext.h"
#include "revng/Pipeline/LLVMKind.h"
#include "revng/Pipes/RevngPasses.h"
#include "revng/Pipes/TaggedFunctionKind.h"
#include "revng/Support/FunctionTags.h"

bool pipeline::ModulePass::runOnModule(llvm::Module &Module) {
  auto &Analysis = getAnalysis<pipeline::LoadExecutionContextPass>();
  ExecutionContext *Ctx = Analysis.get();
  revng_assert(Ctx != nullptr);
  bool Result = false;
  auto Name = Analysis.getContainerName();
  auto *RequestedTargets = &Ctx->getCurrentRequestedTargets()[Name];
  Result = run(Module, *RequestedTargets);
  Ctx->getContext().pushReadFields();
  for (auto &Target : *RequestedTargets)
    Ctx->commit(Target, Analysis.getContainerName());

  Ctx->getContext().popReadFields();
  Ctx->getContext().clearAndResume();

  return Result;
}

bool pipeline::detail::runOnModule(llvm::Module &Module,
                                   FunctionPassImpl &Pipe) {
  auto &Analysis = Pipe.getAnalysis<pipeline::LoadExecutionContextPass>();
  ExecutionContext *Ctx = Analysis.get();
  revng_assert(Ctx != nullptr);
  auto &ModelWrapper = Pipe.getAnalysis<LoadModelWrapperPass>().get();
  bool Result = Pipe.prologue(Module, *ModelWrapper.getReadOnlyModel());

  using Type = revng::kinds::TaggedFunctionKind;
  auto ToIterOn = Type::getFunctionsAndCommit(*Ctx,
                                              Module,
                                              Analysis.getContainerName());
  for (const auto &[First, Second] : ToIterOn) {
    Result = Pipe.runOnFunction(*Second, *First) or Result;
  }

  return Result;
}

void pipeline::detail::getAnalysisUsageImpl(llvm::AnalysisUsage &AU) {
  AU.addRequired<LoadModelWrapperPass>();
  AU.addRequired<pipeline::LoadExecutionContextPass>();
}
