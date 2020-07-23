//
// Copyright rev.ng Srls. See LICENSE.md for details.
//

#include "llvm/ADT/STLExtras.h"
#include "llvm/IR/Function.h"

#include "clang/AST/ASTContext.h"

#include "revng/Support/Assert.h"

#include "DecompilationHelpers.h"
#include "IRASTTypeTranslation.h"

namespace clang {
class TranslationUnitDecl;
} // end namespace clang

void DeclCreator::createTypeDeclsForFunctionPrototype(clang::ASTContext &Ctx,
                                                      llvm::Function *F) {
  clang::TranslationUnitDecl &TUDecl = *Ctx.getTranslationUnitDecl();

  // Create return type
  getOrCreateFunctionRetType(F, Ctx, TUDecl);

  // Create argument types
  for (const auto &[ArgTy, Arg] :
       llvm::zip_first(F->getFunctionType()->params(), F->args())) {
    // For now we don't handle function prototypes with struct arguments.
    // In principle, we expect to never need it, because in assembly arguments
    // are passed to functions by means of registers, that in the decompiled C
    // code will become scalar type.
    revng_assert(not isa<llvm::StructType>(ArgTy));
    revng_assert(not ArgTy->isVoidTy());
    getOrCreateQualType(ArgTy, &Arg, Ctx, TUDecl);
  }
}
