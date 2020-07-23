#pragma once

//
// Copyright rev.ng Srls. See LICENSE.md for details.
//

#include "llvm/ADT/SmallVector.h"

#include "clang/Frontend/ASTConsumers.h"
#include "clang/Frontend/FrontendAction.h"

namespace llvm {
class Function;
} // namespace llvm

class IRASTTypeTranslator;

namespace clang {

class CompilerInstance;

namespace tooling {

class TypeDeclCreationAction : public ASTFrontendAction {

public:
  TypeDeclCreationAction(llvm::Function &F, IRASTTypeTranslator &TT) :
    F(F), TypeTranslator(TT) {}

public:
  std::unique_ptr<ASTConsumer> newASTConsumer();

  virtual std::unique_ptr<ASTConsumer>
  CreateASTConsumer(CompilerInstance &, llvm::StringRef) override;

private:
  llvm::Function &F;
  IRASTTypeTranslator &TypeTranslator;
};

} // end namespace tooling

inline std::unique_ptr<ASTConsumer>
CreateTypeDeclCreator(llvm::Function &F, IRASTTypeTranslator &TT) {
  return tooling::TypeDeclCreationAction(F, TT).newASTConsumer();
}

} // end namespace clang
