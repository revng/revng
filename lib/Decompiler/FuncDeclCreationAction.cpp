//
// Copyright rev.ng Srls. See LICENSE.md for details.
//

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/Module.h"

#include "revng/Support/Assert.h"

#include "FuncDeclCreationAction.h"

#include "DecompilationHelpers.h"
#include "IRASTTypeTranslation.h"
#include "Mangling.h"

using namespace llvm;

namespace clang {
namespace tooling {

class FuncDeclCreator : public ASTConsumer {
public:
  explicit FuncDeclCreator(llvm::Function &F,
                           IRASTTypeTranslator &TT,
                           const dla::ValueLayoutMap *VL) :
    TheF(F), TypeTranslator(TT), ValueLayouts(VL) {}

  virtual void HandleTranslationUnit(ASTContext &Context) override;

protected:
  FunctionDecl *createFunDecl(ASTContext &Context, Function *F);

private:
  llvm::Function &TheF;
  IRASTTypeTranslator &TypeTranslator;
  const dla::ValueLayoutMap *ValueLayouts;
};

FunctionDecl *FuncDeclCreator::createFunDecl(ASTContext &Context, Function *F) {

  TranslationUnitDecl *TUDecl = Context.getTranslationUnitDecl();

  const llvm::FunctionType *FType = F->getFunctionType();

  if (ValueLayouts) {
    auto FItBegin = ValueLayouts->lower_bound(dla::LayoutTypePtr(F, 0));
    auto FItEnd = ValueLayouts->upper_bound(dla::LayoutTypePtr(F));
    // If there is a range of ValueLayots that are indexed by F, they represent
    // the return value of F.
    if (FItBegin != FItEnd) {
      if (std::next(FItBegin) == FItEnd) {
        // F returns a scalar type
        dla::Layout *PointedLayout = FItBegin->second;
        revng_assert(PointedLayout);
      } else {
        // F returns a struct
        for (auto &[_, PointedLayout] : llvm::make_range(FItBegin, FItEnd)) {
          revng_assert(PointedLayout);
        }
      }
    }
  }

  llvm::Type *RetTy = FType->getReturnType();
  QualType RetType = TypeTranslator.getOrCreateQualType(RetTy,
                                                        F,
                                                        Context,
                                                        *TUDecl);

  SmallVector<QualType, 4> ArgTypes = {};
  revng_assert(FType->getNumParams() == F->arg_size());
  for (const auto &[T, Arg] : llvm::zip_first(FType->params(), F->args())) {
    revng_assert(T == Arg.getType());
    // In function declarations all pointers parameters are void *.
    // This is a temporary workaround to reduce warnings
    QualType ArgType = Context.VoidPtrTy;
    if (not isa<llvm::PointerType>(T))
      ArgType = TypeTranslator.getOrCreateQualType(T, &Arg, Context, *TUDecl);
    ArgTypes.push_back(ArgType);
  }
  const bool HasNoParams = ArgTypes.empty();
  const bool IsVariadic = FType->isVarArg();
  if (HasNoParams and not IsVariadic)
    ArgTypes.push_back(Context.VoidTy);

  // Check if the declaration we are tring to emit corresponds to a variadic
  // function, and in that case emit the correct corresponding clang function
  // declaration
  using ExtProtoInfo = FunctionProtoType::ExtProtoInfo;
  ExtProtoInfo ProtoInfo = ExtProtoInfo();
  if (IsVariadic) {
    ProtoInfo.Variadic = true;
  }

  QualType FDeclType = Context.getFunctionType(RetType, ArgTypes, ProtoInfo);

  const llvm::StringRef FName = F->getName();
  revng_assert(not FName.empty());
  IdentifierInfo &FunId = Context.Idents.get(makeCIdentifier(FName));
  StorageClass FunStorage = (F != &TheF) ? StorageClass::SC_Static :
                                           StorageClass::SC_Extern;

  FunctionDecl *NewFDecl = FunctionDecl::Create(Context,
                                                TUDecl,
                                                {},
                                                {},
                                                &FunId,
                                                FDeclType,
                                                nullptr,
                                                FunStorage);

  auto ParmDecls = SmallVector<ParmVarDecl *, 4>(ArgTypes.size(), nullptr);
  if (HasNoParams) {
    revng_assert(ArgTypes.size() == 1 and ArgTypes[0] == Context.VoidTy);
    ParmVarDecl *P = ParmVarDecl::Create(Context,
                                         NewFDecl,
                                         {},
                                         {},
                                         nullptr /* Parameter Identifier */,
                                         ArgTypes[0],
                                         nullptr,
                                         StorageClass::SC_None,
                                         nullptr);
    P->setScopeInfo(0, 0);
    ParmDecls[0] = P;
  } else {
    revng_assert(not ArgTypes.empty());
    revng_assert(F->arg_size() == ArgTypes.size());
    for (auto &Group : llvm::enumerate(F->args())) {
      QualType ArgTy = ArgTypes[Group.index()];
      const std::string ParamName = Group.value().hasName() ?
                                      Group.value().getName().str() :
                                      std::string("param_")
                                        + std::to_string(Group.index());
      IdentifierInfo *ParmId = &Context.Idents.get(makeCIdentifier(ParamName));
      ParmVarDecl *P = ParmVarDecl::Create(Context,
                                           NewFDecl,
                                           {},
                                           {},
                                           ParmId,
                                           ArgTy,
                                           nullptr,
                                           StorageClass::SC_None,
                                           nullptr);
      P->setScopeInfo(0, Group.index());
      ParmDecls[Group.index()] = P;
    }
  }

  NewFDecl->setParams(ParmDecls);
  return NewFDecl;
}

void FuncDeclCreator::HandleTranslationUnit(ASTContext &Context) {
  llvm::Module &M = *TheF.getParent();

  std::set<Function *> Called = getDirectlyCalledFunctions(TheF);
  Called.erase(&TheF);
  // we need abort for decompiling UnreachableInst
  Called.insert(M.getFunction("abort"));
  for (Function *F : Called) {
    const llvm::StringRef FName = F->getName();
    revng_assert(not FName.empty());
    FunctionDecl *NewFDecl = createFunDecl(Context, F);
    TypeTranslator.FunctionDecls[F] = NewFDecl;
  }

  const llvm::StringRef FName = TheF.getName();
  revng_assert(not FName.empty());
  revng_assert(TheF.getMetadata("revng.func.entry"));
  // This is actually a definition, because the isolated function need will
  // be fully decompiled and it needs a body.
  // This definition starts as a declaration that is than inflated by the
  // ASTBuildAnalysis.
  FunctionDecl *NewFDecl = createFunDecl(Context, &TheF);
  TypeTranslator.FunctionDecls[&TheF] = NewFDecl;
}

std::unique_ptr<ASTConsumer> FuncDeclCreationAction::newASTConsumer() {
  return std::make_unique<FuncDeclCreator>(F, TypeTranslator, ValueLayouts);
}

std::unique_ptr<ASTConsumer>
FuncDeclCreationAction::CreateASTConsumer(CompilerInstance &, llvm::StringRef) {
  return newASTConsumer();
}

} // end namespace tooling
} // end namespace clang
