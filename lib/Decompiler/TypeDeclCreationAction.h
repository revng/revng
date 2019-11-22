#ifndef REVNGC_TYPEDECLCREATIONACTION_H
#define REVNGC_TYPEDECLCREATIONACTION_H

// LLVM includes
#include <llvm/ADT/SmallVector.h>

// clang includes
#include <clang/Frontend/ASTConsumers.h>
#include <clang/Frontend/FrontendAction.h>

namespace llvm {
class Function;
} // namespace llvm

namespace clang {

class CompilerInstance;

namespace tooling {

class TypeDeclCreationAction : public ASTFrontendAction {

public:
  using TypeDeclMap = std::map<const llvm::Type *, clang::TypeDecl *>;
  using FieldDeclMap = std::map<clang::TypeDecl *,
                                SmallVector<clang::FieldDecl *, 8>>;

public:
  TypeDeclCreationAction(llvm::Function &F,
                         TypeDeclMap &TDecls,
                         FieldDeclMap &FieldDecls) :
    F(F), TypeDecls(TDecls), FieldDecls(FieldDecls) {}

public:
  std::unique_ptr<ASTConsumer> newASTConsumer();

  virtual std::unique_ptr<ASTConsumer>
  CreateASTConsumer(CompilerInstance &, llvm::StringRef) override;

private:
  llvm::Function &F;
  TypeDeclMap &TypeDecls;
  FieldDeclMap &FieldDecls;
};

} // end namespace tooling

inline std::unique_ptr<ASTConsumer>
CreateTypeDeclCreator(llvm::Function &F,
                      tooling::TypeDeclCreationAction::TypeDeclMap &TypeDecls,
                      tooling::TypeDeclCreationAction::FieldDeclMap &FldDecls) {
  using namespace tooling;
  return TypeDeclCreationAction(F, TypeDecls, FldDecls).newASTConsumer();
}

} // end namespace clang

#endif // REVNGC_TYPEDECLCREATIONACTION_H
