//
// Copyright rev.ng Srls. See LICENSE.md for details.
//

#include "llvm/Pass.h"

#include "revng/Support/FunctionTags.h"

#include "revng-c/MarkForSerialization/MarkAnalysis.h"
#include "revng-c/RestructureCFGPass/RestructureCFG.h"

#include "BeautifyGHAST.h"

struct BeautifyGHASTPass : public llvm::FunctionPass {
  static char ID;

  BeautifyGHASTPass() : llvm::FunctionPass(ID) {}

  void getAnalysisUsage(llvm::AnalysisUsage &AU) const override {
    AU.setPreservesAll();
    AU.addRequired<RestructureCFG>();
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
  if (not FunctionTags::TagsSet::from(&F).contains(FunctionTags::Lifted))
    return false;

  // Get the Abstract Syntax Tree of the restructured code.
  auto &RestructureCFGAnalysis = getAnalysis<RestructureCFG>();
  ASTTree &GHAST = RestructureCFGAnalysis.getAST();

  // Get information about which instructions are marked to be serialized.
  // Mark instructions for serialization, and write the results in ToSerialize
  SerializationMap ToSerialize = {};
  MarkAnalysis::Analysis</* IgnoreDuplicatedUses */ true> Mark(F,
                                                               {},
                                                               ToSerialize);
  Mark.initialize();
  Mark.run();

  beautifyAST(F, GHAST, ToSerialize);

  return false;
}
