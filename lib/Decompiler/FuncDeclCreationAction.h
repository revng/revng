#ifndef REVNGC_FUNCTIONDECLCREATIONACTION_H
#define REVNGC_FUNCTIONDECLCREATIONACTION_H

//
// Copyright rev.ng Srls. See LICENSE.md for details.
//

// clang includes
#include <llvm/ADT/SmallVector.h>

// clang includes
#include <clang/Frontend/ASTConsumers.h>
#include <clang/Frontend/FrontendAction.h>

namespace llvm {
class Function;
class Type;
} // namespace llvm

namespace clang {

class CompilerInstance;
class FieldDecl;
class TypeDecl;

namespace tooling {

class FuncDeclCreationAction : public ASTFrontendAction {

public:
  using FunctionsMap = std::map<llvm::Function *, clang::FunctionDecl *>;
  using TypeDeclMap = std::map<const llvm::Type *, clang::TypeDecl *>;
  using FieldDeclMap = std::map<clang::TypeDecl *,
                                llvm::SmallVector<clang::FieldDecl *, 8>>;

public:
  FuncDeclCreationAction(llvm::Function &F,
                         FunctionsMap &FDecls,
                         TypeDeclMap &TDecls,
                         FieldDeclMap &FieldDecls) :
    F(F), FunctionDecls(FDecls), TypeDecls(TDecls), FieldDecls(FieldDecls) {}

public:
  std::unique_ptr<ASTConsumer> newASTConsumer();

  virtual std::unique_ptr<ASTConsumer>
  CreateASTConsumer(CompilerInstance &, llvm::StringRef) override;

private:
  llvm::Function &F;
  FunctionsMap &FunctionDecls;
  TypeDeclMap &TypeDecls;
  FieldDeclMap &FieldDecls;
};

} // end namespace tooling

inline std::unique_ptr<ASTConsumer>
CreateFuncDeclCreator(llvm::Function &F,
                      tooling::FuncDeclCreationAction::FunctionsMap &FunDecls,
                      tooling::FuncDeclCreationAction::TypeDeclMap &TDecls,
                      tooling::FuncDeclCreationAction::FieldDeclMap &FldDecls) {
  using namespace tooling;
  return FuncDeclCreationAction(F, FunDecls, TDecls, FldDecls).newASTConsumer();
}

} // end namespace clang

#endif // REVNGC_FUNCTIONDECLCREATIONACTION_H
