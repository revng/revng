//
// Copyright (c) rev.ng Srls. See LICENSE.md for details.
//

#include "revng/Model/LoadModelPass.h"

#include "revng-c/HeadersGeneration/ModelToHeader.h"
#include "revng-c/HeadersGeneration/ModelToHeaderPass.h"

llvm::cl::opt<std::string> TypesHeaderName("types-header-name",
                                           llvm::cl::cat(MainCategory),
                                           llvm::cl::Optional,
                                           llvm::cl::init("./revng-types.h"),
                                           llvm::cl::desc("Path of the file "
                                                          "where type "
                                                          "declarations will "
                                                          "be printed."));

bool ModelToHeaderPass::runOnModule(llvm::Module &) {
  auto &ModelPass = getAnalysis<LoadModelWrapperPass>().get();
  const model::Binary &Model = *ModelPass.getReadOnlyModel();

  std::error_code EC;
  llvm::raw_fd_ostream Header(TypesHeaderName, EC);
  if (EC)
    revng_abort(EC.message().c_str());

  dumpModelToHeader(Model, Header);

  return false;
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
