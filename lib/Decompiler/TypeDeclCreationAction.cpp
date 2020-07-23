//
// Copyright rev.ng Srls. See LICENSE.md for details.
//

#include "llvm/IR/Module.h"

#include "revng/Support/Assert.h"

#include "TypeDeclCreationAction.h"

#include "DecompilationHelpers.h"
#include "IRASTTypeTranslation.h"

using namespace llvm;

namespace clang {
namespace tooling {

using TypePair = IRASTTypeTranslator::TypeDeclMap::value_type;

static std::vector<TypePair>
createTypeDecls(ASTContext &Context,
                TranslationUnitDecl *TUDecl,
                IRASTTypeTranslator &TypeTranslator,
                Function *F) {
  std::vector<TypePair> Result;
  const llvm::FunctionType *FType = F->getFunctionType();

  if (llvm::Type *RetTy = dyn_cast<llvm::StructType>(FType->getReturnType())) {

    TypeTranslator.getOrCreateQualType(RetTy, F, Context, *TUDecl);

    revng_assert(TypeTranslator.TypeDecls.count(RetTy));
    Result.push_back(*TypeTranslator.TypeDecls.find(RetTy));
  }

  for (const llvm::Type *T : FType->params()) {
    // For now we don't handle function prototypes with struct arguments
    revng_assert(not isa<llvm::StructType>(T));
  }

  return Result;
}

class TypeDeclCreator : public ASTConsumer {
public:
  explicit TypeDeclCreator(llvm::Function &F, IRASTTypeTranslator &TT) :
    TheF(F), TypeTranslator(TT) {}

  virtual void HandleTranslationUnit(ASTContext &C) override;

private:
  llvm::Function &TheF;
  IRASTTypeTranslator &TypeTranslator;
};

void TypeDeclCreator::HandleTranslationUnit(ASTContext &C) {
  llvm::Module &M = *TheF.getParent();
  TranslationUnitDecl *TUDecl = C.getTranslationUnitDecl();

  std::set<Function *> Called = getDirectlyCalledFunctions(TheF);
  // we need TheF to properly declare its return type if needed
  Called.insert(&TheF);
  // we need abort for decompiling UnreachableInst
  Called.insert(M.getFunction("abort"));
  for (Function *F : Called) {
    const llvm::StringRef FName = F->getName();
    revng_assert(not FName.empty());
    std::vector<TypePair> NewTypeDecls = createTypeDecls(C,
                                                         TUDecl,
                                                         TypeTranslator,
                                                         F);
    for (TypePair &TD : NewTypeDecls)
      TypeTranslator.TypeDecls.insert(TD);
  }
}

std::unique_ptr<ASTConsumer> TypeDeclCreationAction::newASTConsumer() {
  return std::make_unique<TypeDeclCreator>(F, TypeTranslator);
}

std::unique_ptr<ASTConsumer>
TypeDeclCreationAction::CreateASTConsumer(CompilerInstance &, llvm::StringRef) {
  return newASTConsumer();
}

} // end namespace tooling
} // end namespace clang
