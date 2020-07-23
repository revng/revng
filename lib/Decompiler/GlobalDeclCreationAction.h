#pragma once

//
// Copyright rev.ng Srls. See LICENSE.md for details.
//

#include "llvm/ADT/SmallVector.h"

#include "clang/Frontend/ASTConsumers.h"
#include "clang/Frontend/FrontendAction.h"

namespace llvm {
class Function;
class GlobalVariable;
class Type;
class Function;
} // namespace llvm

namespace IR2AST {
class StmtBuilder;
} // namespace IR2AST

class IRASTTypeTranslator;

namespace clang {

class CompilerInstance;
class FieldDecl;
class TypeDecl;

namespace tooling {

class GlobalDeclCreationAction : public ASTFrontendAction {

public:
  GlobalDeclCreationAction(llvm::Function &F,
                           IRASTTypeTranslator &TT,
                           IR2AST::StmtBuilder &ASTBldr) :
    TheF(F), TypeTranslator(TT), ASTBuilder(ASTBldr) {}

public:
  std::unique_ptr<ASTConsumer> newASTConsumer();

  virtual std::unique_ptr<ASTConsumer>
  CreateASTConsumer(CompilerInstance &, llvm::StringRef) override;

private:
  llvm::Function &TheF;
  IRASTTypeTranslator &TypeTranslator;
  IR2AST::StmtBuilder &ASTBuilder;
};

} // end namespace tooling

inline std::unique_ptr<ASTConsumer>
CreateGlobalDeclCreator(llvm::Function &F,
                        IRASTTypeTranslator &TT,
                        IR2AST::StmtBuilder &Bldr) {
  return tooling::GlobalDeclCreationAction(F, TT, Bldr).newASTConsumer();
}

} // end namespace clang
