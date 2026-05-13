//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/IR/Function.h"
#include "llvm/IR/Module.h"
#include "llvm/Pass.h"

#include "revng/InlineHelpers/DeleteHelperBodies.h"
#include "revng/Support/IRHelpers.h"

#include "PostInlineHelpersVerifyPass.h"

using namespace llvm;

// Drop the body of every `revng_inline`-tagged function in `M`, turning each
// one back into a declaration.
static void deleteHelperBodies(llvm::Module &M) {
  for (llvm::Function &F : M) {
    if (F.getSection() == InlineHelpersSection)
      deleteOnlyBody(F);
  }
}

char DeleteHelperBodiesPass::ID = 0;

using Register = RegisterPass<DeleteHelperBodiesPass>;
static Register
  X("delete-helper-bodies", "Delete Helper Bodies Pass", true, true);

bool DeleteHelperBodiesPass::runOnModule(llvm::Module &M) {
  deleteHelperBodies(M);

  // Verify that no `getelementptr` instruction was dragged into an Isolated
  // function by the inlining performed by the upstream `-inline-helpers`
  // pass. The downstream pipeline relies on this invariant.
  // TODO: convert this from a pass to a free-standing function
  PostInlineHelpersVerifyPass{}.runOnModule(M);

  return true;
}
