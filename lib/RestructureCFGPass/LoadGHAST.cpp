//
// Copyright rev.ng Labs Srl. See LICENSE.md for details.
//
//

#include "llvm/Pass.h"

#include "revng-c/RestructureCFGPass/LoadGHAST.h"

using namespace llvm;

char LoadGHASTWrapperPass::ID;

using Reg = llvm::RegisterPass<LoadGHASTWrapperPass>;
static Reg X("load-ghast", "Initialize the GHAST", true, true);
