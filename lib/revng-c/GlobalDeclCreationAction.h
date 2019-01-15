#ifndef REVNGC_GLOBALDECLCREATIONACTION_H
#define REVNGC_GLOBALDECLCREATIONACTION_H

#include <clang/Frontend/ASTConsumers.h>
#include <clang/Frontend/FrontendAction.h>

namespace llvm {
class Module;
class GlobalVariable;
} // namespace llvm

namespace clang {

class CompilerInstance;

namespace tooling {

class GlobalDeclCreationAction : public ASTFrontendAction {

public:
  using GlobalsMap = std::map<const llvm::GlobalVariable *, clang::VarDecl *>;

public:
  GlobalDeclCreationAction(llvm::Module &M, GlobalsMap &Map) :
    M(M),
    GlobalVarAST(Map) {}

public:
  std::unique_ptr<ASTConsumer> newASTConsumer();

  virtual std::unique_ptr<ASTConsumer>
  CreateASTConsumer(CompilerInstance &, llvm::StringRef) override {
    return newASTConsumer();
  }

private:
  llvm::Module &M;
  GlobalsMap &GlobalVarAST;
};

} // end namespace tooling

inline std::unique_ptr<ASTConsumer>
CreateGlobalDeclCreator(llvm::Module &Module,
                        tooling::GlobalDeclCreationAction::GlobalsMap &Map) {
  return tooling::GlobalDeclCreationAction(Module, Map).newASTConsumer();
}

} // end namespace clang

#endif // REVNGC_GLOBALDECLCREATIONACTION_H
