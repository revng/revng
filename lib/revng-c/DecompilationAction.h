#ifndef REVNGC_DECOMPILATIONACTION_H
#define REVNGC_DECOMPILATIONACTION_H

#include <clang/Frontend/ASTConsumers.h>
#include <clang/Frontend/FrontendAction.h>
#include <cstdio>

namespace clang {

class CompilerInstance;

namespace tooling {

class DecompilationAction : public ASTFrontendAction {

public:
  DecompilationAction(llvm::Module &M,
                      const llvm::Function *F,
                      std::unique_ptr<llvm::raw_ostream> O) :
    M(M),
    F(F),
    O(std::move(O)) {}

public:
  std::unique_ptr<ASTConsumer> newASTConsumer();

  virtual std::unique_ptr<ASTConsumer>
  CreateASTConsumer(CompilerInstance &, llvm::StringRef) override {
    return newASTConsumer();
  }

private:
  llvm::Module &M;
  const llvm::Function *F;
  std::unique_ptr<llvm::raw_ostream> O;
};

} // end namespace tooling

} // namespace clang

#endif // REVNGC_DECOMPILATIONACTION_H
