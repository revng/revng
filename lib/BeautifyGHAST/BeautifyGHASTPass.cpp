//
// Copyright rev.ng Labs Srl. See LICENSE.md for details.
//

#include "llvm/Pass.h"

#include "revng/Support/FunctionTags.h"

#include "revng-c/BeautifyGHAST/BeautifyGHAST.h"
#include "revng-c/RestructureCFGPass/LoadGHAST.h"

struct BeautifyGHASTPass : public llvm::FunctionPass {
  static char ID;

  BeautifyGHASTPass() : llvm::FunctionPass(ID) {}

  void getAnalysisUsage(llvm::AnalysisUsage &AU) const override {
    AU.setPreservesAll();
    AU.addRequired<LoadGHASTWrapperPass>();
  }

  bool runOnFunction(llvm::Function &F) override;
};

char BeautifyGHASTPass::ID = 0;

using Register = llvm::RegisterPass<BeautifyGHASTPass>;
static Register X("beautify-ghast", "GHAST Beautyfication Pass", false, false);

bool BeautifyGHASTPass::runOnFunction(llvm::Function &F) {
  // Skip functions without body
  if (F.isDeclaration())
    return false;

  // Skip functions that are not lifted
  if (not FunctionTags::TagsSet::from(&F).contains(FunctionTags::Isolated))
    return false;

  // Get and beautify the Abstract Syntax Tree
  beautifyAST(F, getAnalysis<LoadGHASTWrapperPass>().getGHAST(F));

  return false;
}
