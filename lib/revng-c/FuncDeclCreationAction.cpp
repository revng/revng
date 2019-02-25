#include <llvm/IR/Module.h>

#include <revng/Support/Assert.h>

#include "FuncDeclCreationAction.h"

#include "DecompilationHelpers.h"
#include "IRASTTypeTranslation.h"
#include "Mangling.h"

using namespace llvm;

namespace clang {
namespace tooling {

using FunctionsMap = FuncDeclCreationAction::FunctionsMap;

static FunctionDecl *createFunDecl(ASTContext &Context,
                                   TranslationUnitDecl *TUDecl,
                                   Function *F,
                                   bool hasBody) {

  const llvm::FunctionType *FType = F->getFunctionType();

  llvm::Type *RetTy = FType->getReturnType();
  QualType RetType = IRASTTypeTranslation::getQualType(RetTy, Context);

  SmallVector<QualType, 4> VoidArgsTy(1, Context.VoidTy);
  SmallVector<QualType, 4> ArgTypes = {};
  for (const llvm::Type *T : FType->params()) {
    // In function declarations all pointers parameters are void *.
    // This is a temporary workaround to reduce warnings
    QualType ArgType = Context.VoidPtrTy;
    if (not isa<llvm::PointerType>(T))
      ArgType = IRASTTypeTranslation::getQualType(T, Context);
    ArgTypes.push_back(ArgType);
  }
  const bool HasNoParams = ArgTypes.empty();
  if (HasNoParams)
    ArgTypes.push_back(Context.VoidTy);

  using ExtProtoInfo = FunctionProtoType::ExtProtoInfo;
  QualType FDeclType = Context.getFunctionType(RetType,
                                               ArgTypes,
                                               ExtProtoInfo());

  const llvm::StringRef FName = F->getName();
  revng_assert(not FName.empty());
  IdentifierInfo &FunId = Context.Idents.get(makeCIdentifier(FName));
  StorageClass FunStorage = hasBody ? StorageClass::SC_Static :
                                      StorageClass::SC_Extern;

  FunctionDecl *NewFDecl = FunctionDecl::Create(Context,
                                                TUDecl,
                                                {},
                                                {},
                                                &FunId,
                                                FDeclType,
                                                nullptr,
                                                FunStorage);
  TUDecl->addDecl(NewFDecl);
  int N = 0;
  auto ParmDecls = SmallVector<ParmVarDecl *, 4>(ArgTypes.size(), nullptr);
  for (QualType ArgTy : ArgTypes) {
    int ParamIdx = N++;
    IdentifierInfo *ParmId = nullptr;
    if (not HasNoParams)
      ParmId = &Context.Idents.get("param_" + std::to_string(ParamIdx));
    ParmVarDecl *P = ParmVarDecl::Create(Context,
                                         NewFDecl,
                                         {},
                                         {},
                                         ParmId,
                                         ArgTy,
                                         nullptr,
                                         StorageClass::SC_None,
                                         nullptr);
    P->setScopeInfo(0, ParamIdx);
    ParmDecls[ParamIdx] = P;
  }
  NewFDecl->setParams(ParmDecls);
  return NewFDecl;
}

class FuncDeclCreator : public ASTConsumer {
public:
  explicit FuncDeclCreator(llvm::Module &M, FunctionsMap &Decls) :
    M(M),
    FunctionDecls(Decls) {}

  virtual void HandleTranslationUnit(ASTContext &Context) override {
    TranslationUnitDecl *TUDecl = Context.getTranslationUnitDecl();
    std::set<Function *> IsolatedFunctions = getIsolatedFunctions(M);

    std::set<Function *> Called = getDirectlyCalledFunctions(IsolatedFunctions);
    // we need abort for decompiling UnreachableInst
    Called.insert(M.getFunction("abort"));
    for (Function *F : Called) {
      const llvm::StringRef FName = F->getName();
      revng_assert(not FName.empty());
      bool IsIsolated = FName.startswith("bb.");
      FunctionDecl *NewFDecl = createFunDecl(Context, TUDecl, F, IsIsolated);
      FunctionDecls[F] = NewFDecl;
    }

    for (Function *F : IsolatedFunctions) {
      const llvm::StringRef FName = F->getName();
      revng_assert(not FName.empty());
      revng_assert(FName.startswith("bb."));
      // This is actually a definition, because the isolated functions need
      // will be fully decompiled and they need a body.
      // We still emit the forward declarations for all the prototypes, and
      // we emit a full definition for the isolated functions.
      // This definition starts as a declaration that is than inflated by the
      // ASTBuildAnalysis.
      FunctionDecl *NewFDecl = createFunDecl(Context, TUDecl, F, true);
      FunctionDecls[F] = NewFDecl;
    }
  }

private:
  llvm::Module &M;
  FunctionsMap &FunctionDecls;
};

std::unique_ptr<ASTConsumer> FuncDeclCreationAction::newASTConsumer() {
  return std::make_unique<FuncDeclCreator>(M, FunctionDecls);
}

} // end namespace tooling
} // end namespace clang
