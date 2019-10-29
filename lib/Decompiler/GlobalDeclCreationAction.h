#ifndef REVNGC_GLOBALDECLCREATIONACTION_H
#define REVNGC_GLOBALDECLCREATIONACTION_H

// LLVM includes
#include <llvm/ADT/SmallVector.h>

// clang includes
#include <clang/Frontend/ASTConsumers.h>
#include <clang/Frontend/FrontendAction.h>

namespace llvm {
class GlobalVariable;
class Type;
} // namespace llvm

namespace clang {

class CompilerInstance;
class FieldDecl;
class TypeDecl;

namespace tooling {

class GlobalDeclCreationAction : public ASTFrontendAction {

public:
  using GlobalsMap = std::map<const llvm::GlobalVariable *, clang::VarDecl *>;
  using TypeDeclMap = std::map<const llvm::Type *, clang::TypeDecl *>;
  using FieldDeclMap = std::map<clang::TypeDecl *,
                                llvm::SmallVector<clang::FieldDecl *, 8>>;

public:
  GlobalDeclCreationAction(llvm::Function &F,
                           GlobalsMap &GMap,
                           TypeDeclMap &TDecls,
                           FieldDeclMap &FieldDecls) :
    TheF(F),
    GlobalVarAST(GMap),
    TypeDecls(TDecls),
    FieldDecls(FieldDecls) {}

public:
  std::unique_ptr<ASTConsumer> newASTConsumer();

  virtual std::unique_ptr<ASTConsumer>
  CreateASTConsumer(CompilerInstance &, llvm::StringRef) override;

private:
  llvm::Function &TheF;
  GlobalsMap &GlobalVarAST;
  TypeDeclMap &TypeDecls;
  FieldDeclMap &FieldDecls;
};

} // end namespace tooling

inline std::unique_ptr<ASTConsumer>
CreateGlobalDeclCreator(llvm::Function &F,
                        tooling::GlobalDeclCreationAction::GlobalsMap &Map,
                        tooling::GlobalDeclCreationAction::TypeDeclMap &TDecl,
                        tooling::GlobalDeclCreationAction::FieldDeclMap &FldD) {
  using namespace tooling;
  return GlobalDeclCreationAction(F, Map, TDecl, FldD).newASTConsumer();
}

} // end namespace clang

#endif // REVNGC_GLOBALDECLCREATIONACTION_H
