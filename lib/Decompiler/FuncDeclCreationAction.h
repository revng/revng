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
} // namespace llvm

class IRASTTypeTranslator;

namespace clang {

class CompilerInstance;

namespace tooling {

class FuncDeclCreationAction : public ASTFrontendAction {

public:
  FuncDeclCreationAction(llvm::Function &F,
                         IRASTTypeTranslator &TT,
                         const dla::ValueLayoutMap *VL) :
    F(F), TypeTranslator(TT), ValueLayouts(VL) {}

public:
  std::unique_ptr<ASTConsumer> newASTConsumer();

  virtual std::unique_ptr<ASTConsumer>
  CreateASTConsumer(CompilerInstance &, llvm::StringRef) override;

private:
  llvm::Function &F;
  IRASTTypeTranslator &TypeTranslator;
  const dla::ValueLayoutMap *ValueLayouts;
};

} // end namespace tooling

inline std::unique_ptr<ASTConsumer>
CreateFuncDeclCreator(llvm::Function &F,
                      IRASTTypeTranslator &TT,
                      const dla::ValueLayoutMap *VL) {
  auto FDeclCreator = tooling::FuncDeclCreationAction(F, TT, VL);
  return FDeclCreator.newASTConsumer();
}

} // end namespace clang
