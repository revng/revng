//
// Copyright (c) rev.ng Srls. See LICENSE.md for details.
//

#include "revng-c/Decompiler/DLAPass.h"

char DLAPass::ID = 0;

using Register = llvm::RegisterPass<DLAPass>;
static Register X("dla", "Data Layout Analysis Pass", false, false);

bool DLAPass::runOnModule(llvm::Module &) {
  return true;
}
