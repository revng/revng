//
// Copyright (c) rev.ng Srls. See LICENSE.md for details.
//

#include "revng/Model/LoadModelPass.h"

#include "revng-c/ModelToHeader/ModelToHeaderPass.h"

bool ModelToHeaderPass::runOnModule(llvm::Module &) {
  auto &ModelPass = getAnalysis<LoadModelWrapperPass>().get();
  [[maybe_unused]] const model::Binary &Model = ModelPass.getReadOnlyModel();
  return true;
}

void ModelToHeaderPass::getAnalysisUsage(llvm::AnalysisUsage &AU) const {
  AU.setPreservesAll();
  AU.addRequired<LoadModelWrapperPass>();
}

char ModelToHeaderPass::ID = 0;

using llvm::RegisterPass;
static RegisterPass<ModelToHeaderPass> X("model-to-header",
                                         "Pass that takes the model and prints "
                                         "a C header for all the type "
                                         "declarations, global variable "
                                         "declarations, and function "
                                         "declarations necessary");
