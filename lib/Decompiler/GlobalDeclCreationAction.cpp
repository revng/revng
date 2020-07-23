//
// Copyright rev.ng Srls. See LICENSE.md for details.
//

#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Module.h"

#include "revng/Support/Assert.h"

#include "GlobalDeclCreationAction.h"

#include "ASTBuildAnalysis.h"
#include "DecompilationHelpers.h"
#include "IRASTTypeTranslation.h"
#include "Mangling.h"

namespace clang {
namespace tooling {

class GlobalDeclsCreator : public ASTConsumer {
public:
  explicit GlobalDeclsCreator(llvm::Function &F,
                              IRASTTypeTranslator &TT,
                              IR2AST::StmtBuilder &ASTBldr) :
    TheF(F), TypeTranslator(TT), ASTBuilder(ASTBldr) {}

  virtual void HandleTranslationUnit(ASTContext &Context) override;

private:
  llvm::Function &TheF;
  IRASTTypeTranslator &TypeTranslator;
  IR2AST::StmtBuilder &ASTBuilder;
};

void GlobalDeclsCreator::HandleTranslationUnit(ASTContext &Context) {
  uint64_t UnnamedNum = 0;
  TranslationUnitDecl *TUDecl = Context.getTranslationUnitDecl();
  for (llvm::GlobalVariable *G : getDirectlyUsedGlobals(TheF)) {
    QualType ASTTy = TypeTranslator.getOrCreateQualType(G, Context, *TUDecl);

    std::string VarName = G->getName();
    if (VarName.empty()) {
      llvm::raw_string_ostream Stream(VarName);
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

      llvm::Constant *LLVMInit = G->getInitializer();
      const clang::Type *UnderlyingTy = ASTTy.getTypePtrOrNull();
      if (UnderlyingTy != nullptr and not isa<llvm::ConstantExpr>(LLVMInit)) {
        clang::Expr *Init = nullptr;
        if (UnderlyingTy->isCharType()) {
          uint64_t UniqueInteger = LLVMInit->getUniqueInteger().getZExtValue();
          revng_assert(UniqueInteger < 256);
          Init = new (Context)
            CharacterLiteral(static_cast<unsigned>(UniqueInteger),
                             CharacterLiteral::CharacterKind::Ascii,
                             Context.CharTy,
                             {});
        } else if (UnderlyingTy->isBooleanType()) {
          const llvm::ConstantInt *CInt = cast<llvm::ConstantInt>(LLVMInit);
          uint64_t InitValue = CInt->getValue().getZExtValue();
          llvm::APInt InitVal = LLVMInit->getUniqueInteger();
          auto BoolTy = TypeTranslator.getOrCreateBoolQualType(Context,
                                                               G->getType());
          QualType IntT = Context.IntTy;
          auto Const = llvm::APInt(Context.getIntWidth(IntT), InitValue, true);
          Expr *IntLiteral = IntegerLiteral::Create(Context, Const, IntT, {});
          Init = createCast(BoolTy, IntLiteral, Context);
        } else if (UnderlyingTy->isIntegerType()
                   and not UnderlyingTy->isPointerType()
                   and not UnderlyingTy->isAnyCharacterType()) {

          Init = ASTBuilder.getLiteralFromConstant(LLVMInit);
        }

        if (Init)
          NewVar->setInit(Init);
      }
    }
    TypeTranslator.GlobalDecls[G] = NewVar;
  }
}

std::unique_ptr<ASTConsumer> GlobalDeclCreationAction::newASTConsumer() {
  return std::make_unique<GlobalDeclsCreator>(TheF, TypeTranslator, ASTBuilder);
}

std::unique_ptr<ASTConsumer>
GlobalDeclCreationAction::CreateASTConsumer(CompilerInstance &,
                                            llvm::StringRef) {
  return newASTConsumer();
}

} // end namespace tooling
} // end namespace clang
