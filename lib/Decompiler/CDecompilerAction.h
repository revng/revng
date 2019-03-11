#ifndef REVNGC_DECOMPILATIONACTION_H
#define REVNGC_DECOMPILATIONACTION_H

#include <clang/Frontend/ASTConsumers.h>
#include <clang/Frontend/FrontendAction.h>
#include <cstdio>

class ASTTree;

namespace clang {

class CompilerInstance;

namespace tooling {

class CDecompilerAction : public ASTFrontendAction {

public:
  CDecompilerAction(llvm::Function &F,
                      ASTTree &CombedAST,
                      std::unique_ptr<llvm::raw_ostream> O) :

    F(F),
    CombedAST(CombedAST),
    O(std::move(O)) {}

public:
  std::unique_ptr<ASTConsumer> newASTConsumer();

  virtual std::unique_ptr<ASTConsumer>
  CreateASTConsumer(CompilerInstance &, llvm::StringRef) override {
    return newASTConsumer();
  }

private:
  llvm::Function &F;
  ASTTree &CombedAST;
  std::unique_ptr<llvm::raw_ostream> O;
};

} // end namespace tooling

} // namespace clang

#endif // REVNGC_DECOMPILATIONACTION_H
