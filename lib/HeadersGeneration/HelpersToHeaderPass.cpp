//
// Copyright (c) rev.ng Srls. See LICENSE.md for details.
//

#include "revng/Model/LoadModelPass.h"

#include "revng-c/HeadersGeneration/HelpersToHeader.h"
#include "revng-c/HeadersGeneration/HelpersToHeaderPass.h"

llvm::cl::opt<std::string> HelpersHeaderName("helpers-header-name",
                                             llvm::cl::cat(MainCategory),
                                             llvm::cl::Optional,
                                             llvm::cl::init("./helpers.h"),
                                             llvm::cl::desc("Path of the file "
                                                            "where helper "
                                                            "functions "
                                                            "declarations will "
                                                            "be printed."));

bool HelpersToHeaderPass::runOnModule(llvm::Module &Module) {
  std::error_code EC;
  llvm::raw_fd_ostream Header(HelpersHeaderName, EC);
  if (EC)
    revng_abort(EC.message().c_str());

  dumpHelpersToHeader(Module, Header);

  return false;
}

void HelpersToHeaderPass::getAnalysisUsage(llvm::AnalysisUsage &AU) const {
  AU.setPreservesAll();
}

char HelpersToHeaderPass::ID = 0;

using llvm::RegisterPass;
static RegisterPass<HelpersToHeaderPass> X("helpers-to-header",
                                           "Pass that takes a module and "
                                           "prints a C header with all the "
                                           "QEMU and revng helpers "
                                           "declarations "
                                           "needed by the module");
