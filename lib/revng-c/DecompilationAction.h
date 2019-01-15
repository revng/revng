#ifndef REVNGC_DECOMPILATIONACTION_H
#define REVNGC_DECOMPILATIONACTION_H

#include <clang/Frontend/ASTConsumers.h>
#include <clang/Frontend/FrontendAction.h>

namespace clang {

class CompilerInstance;

namespace tooling {

class DecompilationAction : public ASTFrontendAction {

public:
  DecompilationAction(llvm::Module &M) : M(M) {}

public:
  std::unique_ptr<ASTConsumer> newASTConsumer();

  virtual std::unique_ptr<ASTConsumer>
  CreateASTConsumer(CompilerInstance &, llvm::StringRef) override {
    return newASTConsumer();
  }

private:
  llvm::Module &M;
};

} // end namespace tooling

} // end namespace clang

#endif // REVNGC_DECOMPILATIONACTION_H
