//
// Copyright rev.ng Srls. See LICENSE.md for details.
//

#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/Constants.h"

#include "clang/AST/ASTContext.h"
#include "clang/AST/Expr.h"

#include "revng/Support/Assert.h"

#include "ASTBuildAnalysis.h"
#include "DecompilationHelpers.h"
#include "IRASTTypeTranslation.h"
#include "Mangling.h"

using clang::CharacterLiteral;
using clang::Expr;
using clang::IntegerLiteral;
using clang::QualType;
using llvm::Function;

using IR2AST::StmtBuilder;

void DeclCreator::createGlobalVarDeclUsedByFunction(clang::ASTContext &Context,
                                                    const llvm::Function *TheF,
                                                    StmtBuilder &ASTBuilder) {
  uint64_t UnnamedNum = 0;
  clang::TranslationUnitDecl *TUDecl = Context.getTranslationUnitDecl();
  for (const llvm::GlobalVariable *G : getDirectlyUsedGlobals(*TheF)) {
    QualType ASTTy = getQualType(getOrCreateType(G, Context, *TUDecl));

    std::string VarName = G->getName().str();
    if (VarName.empty()) {
      llvm::raw_string_ostream Stream(VarName);
      Stream << "global_" << UnnamedNum++;
    }
    clang::IdentifierInfo &Id = Context.Idents.get(makeCIdentifier(VarName));
    auto *NewVar = clang::VarDecl::Create(Context,
                                          TUDecl,
                                          {},
                                          {},
                                          &Id,
                                          ASTTy,
                                          nullptr,
                                          clang::StorageClass::SC_None);
    if (G->hasInitializer()) {
      revng_assert(not G->isExternallyInitialized());

      const llvm::Constant *LLVMInit = G->getInitializer();
      const clang::Type *UnderlyingTy = ASTTy.getTypePtrOrNull();
      if (UnderlyingTy != nullptr and not isa<llvm::ConstantExpr>(LLVMInit)) {
        Expr *Init = nullptr;
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
          TypeDeclOrQualType BoolTy = getOrCreateBoolType(Context,
                                                          G->getType());
          QualType IntT = Context.IntTy;
          auto Const = llvm::APInt(Context.getIntWidth(IntT), InitValue, true);
          Expr *IntLiteral = IntegerLiteral::Create(Context, Const, IntT, {});
          Init = createCast(DeclCreator::getQualType(BoolTy),
                            IntLiteral,
                            Context);
        } else if (UnderlyingTy->isIntegerType()
                   and not UnderlyingTy->isPointerType()
                   and not UnderlyingTy->isAnyCharacterType()) {

          Init = ASTBuilder.getLiteralFromConstant(LLVMInit);
        }

        if (Init)
          NewVar->setInit(Init);
      }
    }
    GlobalDecls[G] = NewVar;
  }
}
