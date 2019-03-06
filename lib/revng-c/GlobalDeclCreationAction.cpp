#include <llvm/IR/Constants.h>
#include <llvm/IR/Module.h>

#include <revng/Support/Assert.h>

#include "GlobalDeclCreationAction.h"

#include "DecompilationHelpers.h"
#include "IRASTTypeTranslation.h"
#include "Mangling.h"

using namespace llvm;

namespace clang {
namespace tooling {

using GlobalsMap = GlobalDeclCreationAction::GlobalsMap;

class GlobalDeclsCreator : public ASTConsumer {
public:
  explicit GlobalDeclsCreator(llvm::Function &F, GlobalsMap &Map) :
    TheF(F),
    GlobalVarAST(Map) {}

  virtual void HandleTranslationUnit(ASTContext &Context) override {
    uint64_t UnnamedNum = 0;
    TranslationUnitDecl *TUDecl = Context.getTranslationUnitDecl();
    for (const GlobalVariable *G : getDirectlyUsedGlobals(TheF)) {
      QualType ASTTy = IRASTTypeTranslation::getQualType(G, Context);

      std::string VarName = G->getName();
      if (VarName.empty()) {
        raw_string_ostream Stream(VarName);
        Stream << "global_" << UnnamedNum++;
      }
      IdentifierInfo &Id = Context.Idents.get(makeCIdentifier(VarName));
      VarDecl *NewVar = VarDecl::Create(Context,
                                        TUDecl,
                                        {},
                                        {},
                                        &Id,
                                        ASTTy,
                                        nullptr,
                                        StorageClass::SC_Static);
      if (G->hasInitializer()) {
        revng_assert(not G->isExternallyInitialized());

        const llvm::Constant *LLVMInit = G->getInitializer();
        const clang::Type *UnderlyingTy = ASTTy.getTypePtrOrNull();
        if (UnderlyingTy != nullptr and not isa<llvm::ConstantExpr>(LLVMInit)) {
          clang::Expr *Init = nullptr;
          if (UnderlyingTy->isCharType()) {
            Init = new (Context)
              CharacterLiteral(LLVMInit->getUniqueInteger().getZExtValue(),
                               CharacterLiteral::CharacterKind::Ascii,
                               Context.CharTy,
                               {});
          } else if (UnderlyingTy->isIntegerType()
                     and not UnderlyingTy->isPointerType()
                     and not UnderlyingTy->isAnyCharacterType()) {
            Init = IntegerLiteral::Create(Context,
                                          LLVMInit->getUniqueInteger(),
                                          ASTTy,
                                          {});
          }

          if (Init)
            NewVar->setInit(Init);
        }
      }
      TUDecl->addDecl(NewVar);
      GlobalVarAST[G] = NewVar;
    }
  }

private:
  llvm::Function &TheF;
  GlobalsMap &GlobalVarAST;
};

std::unique_ptr<ASTConsumer> GlobalDeclCreationAction::newASTConsumer() {
  return std::make_unique<GlobalDeclsCreator>(TheF, GlobalVarAST);
}

} // end namespace tooling
} // end namespace clang
