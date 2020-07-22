#pragma once

//
// Copyright rev.ng Srls. See LICENSE.md for details.
//

#include "llvm/ADT/SmallVector.h"

#include "clang/Frontend/ASTConsumers.h"
#include "clang/Frontend/FrontendAction.h"

#include "revng-c/Decompiler/DLALayouts.h"

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
                         FieldDeclMap &FieldDecls,
                         const dla::ValueLayoutMap *VL) :
    F(F),
    FunctionDecls(FDecls),
    TypeDecls(TDecls),
    FieldDecls(FieldDecls),
    ValueLayouts(VL) {}

public:
  std::unique_ptr<ASTConsumer> newASTConsumer();

  virtual std::unique_ptr<ASTConsumer>
  CreateASTConsumer(CompilerInstance &, llvm::StringRef) override;

private:
  llvm::Function &F;
  FunctionsMap &FunctionDecls;
  TypeDeclMap &TypeDecls;
  FieldDeclMap &FieldDecls;
  const dla::ValueLayoutMap *ValueLayouts;
};

} // end namespace tooling

inline std::unique_ptr<ASTConsumer>
CreateFuncDeclCreator(llvm::Function &F,
                      tooling::FuncDeclCreationAction::FunctionsMap &FunDecls,
                      tooling::FuncDeclCreationAction::TypeDeclMap &TDecls,
                      tooling::FuncDeclCreationAction::FieldDeclMap &FldDecls,
                      const dla::ValueLayoutMap *VL) {
  using namespace tooling;
  auto FDeclCreator = FuncDeclCreationAction(F, FunDecls, TDecls, FldDecls, VL);
  return FDeclCreator.newASTConsumer();
}

} // end namespace clang
