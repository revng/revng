// LLVM includes
#include <llvm/IR/Module.h>

// revng includes
#include <revng/Support/Assert.h>

// local includes
#include "DecompilationHelpers.h"
#include "IRASTTypeTranslation.h"
#include "TypeDeclCreationAction.h"

using namespace llvm;

namespace clang {
namespace tooling {

using TypeDeclMap = TypeDeclCreationAction::TypeDeclMap;
using FieldDeclMap = TypeDeclCreationAction::FieldDeclMap;
using TypePair = std::pair<const llvm::Type *const, clang::TypeDecl *>;

static std::vector<TypePair> createTypeDecls(ASTContext &Context,
                                             TranslationUnitDecl *TUDecl,
                                             TypeDeclMap &TypeDecls,
                                             FieldDeclMap &FieldDecls,
                                             Function *F) {
  std::vector<TypePair> Result;
  const llvm::FunctionType *FType = F->getFunctionType();

  if (llvm::Type *RetTy = dyn_cast<llvm::StructType>(FType->getReturnType())) {

    IRASTTypeTranslation::getOrCreateQualType(RetTy,
                                              Context,
                                              *TUDecl,
                                              TypeDecls,
                                              FieldDecls);

    revng_assert(TypeDecls.count(RetTy));
    Result.push_back(*TypeDecls.find(RetTy));
  }

  for (const llvm::Type *T : FType->params()) {
    // For now we don't handle function prototypes with struct arguments
    revng_assert(not isa<llvm::StructType>(T));
  }

  return Result;
}

class TypeDeclCreator : public ASTConsumer {
public:
  explicit TypeDeclCreator(llvm::Function &F,
                           TypeDeclMap &TDecls,
                           FieldDeclMap &FieldDecls) :
    TheF(F),
    TypeDecls(TDecls),
    FieldDecls(FieldDecls) {}

  virtual void HandleTranslationUnit(ASTContext &C) override;

private:
  llvm::Function &TheF;
  TypeDeclMap &TypeDecls;
  FieldDeclMap &FieldDecls;
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
                                                         TypeDecls,
                                                         FieldDecls,
                                                         F);
    for (TypePair &TD : NewTypeDecls)
      TypeDecls.insert(TD);
  }
}

std::unique_ptr<ASTConsumer> TypeDeclCreationAction::newASTConsumer() {
  return std::make_unique<TypeDeclCreator>(F, TypeDecls, FieldDecls);
}

std::unique_ptr<ASTConsumer>
TypeDeclCreationAction::CreateASTConsumer(CompilerInstance &, llvm::StringRef) {
  return newASTConsumer();
}

} // end namespace tooling
} // end namespace clang
