//
// Copyright rev.ng Srls. See LICENSE.md for details.
//

#include <set>

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/Function.h"

#include "clang/AST/ASTContext.h"

#include "revng/Support/Assert.h"
#include "revng/Support/FunctionTags.h"

#include "revng-c/Support/Mangling.h"

#include "DecompilationHelpers.h"
#include "IRASTTypeTranslation.h"

namespace clang {
class TranslationUnitDecl;
} // end namespace clang

using ExtProtoInfo = clang::FunctionProtoType::ExtProtoInfo;
using clang::StorageClass;

clang::FunctionDecl *DeclCreator::createFunDecl(clang::ASTContext &Context,
                                                const llvm::Function *F,
                                                bool IsDefinition) {
  using clang::QualType;

  clang::TranslationUnitDecl *TUDecl = Context.getTranslationUnitDecl();
  QualType RetType = getQualType(getOrCreateType(F, Context, *TUDecl));

  const llvm::FunctionType *FType = F->getFunctionType();
  revng_assert(FType->getNumParams() == F->arg_size());

  llvm::SmallVector<QualType, 4> ArgTypes = {};
  for (const llvm::Argument &Arg : F->args()) {
    QualType ArgType = getQualType(getOrCreateType(&Arg, Context, *TUDecl));
    ArgTypes.push_back(ArgType);
  }

  const bool HasNoParams = ArgTypes.empty();
  const bool IsVariadic = FType->isVarArg();
  if (HasNoParams and not IsVariadic)
    ArgTypes.push_back(Context.VoidTy);

  // Check if the declaration we are tring to emit corresponds to a variadic
  // function, and in that case emit the correct corresponding clang function
  // declaration
  ExtProtoInfo ProtoInfo = ExtProtoInfo();
  if (IsVariadic) {
    ProtoInfo.Variadic = true;
  }

  QualType FDeclType = Context.getFunctionType(RetType, ArgTypes, ProtoInfo);

  const llvm::StringRef FName = F->getName();
  revng_assert(not FName.empty());

  using clang::IdentifierInfo;
  IdentifierInfo &FunId = Context.Idents.get(makeCIdentifier(FName.str()));
  StorageClass FunStorage = IsDefinition ? StorageClass::SC_None :
                                           StorageClass::SC_Extern;

  using clang::FunctionDecl;
  FunctionDecl *NewFDecl = FunctionDecl::Create(Context,
                                                TUDecl,
                                                {},
                                                {},
                                                &FunId,
                                                FDeclType,
                                                nullptr,
                                                FunStorage);

  using clang::ParmVarDecl;
  using ParmVarDeclV = llvm::SmallVector<ParmVarDecl *, 4>;
  ParmVarDeclV ParmDecls(ArgTypes.size(), nullptr);

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
    for (auto &Group : llvm::enumerate(llvm::zip_first(F->args(), ArgTypes))) {
      const auto &[Arg, ArgQualTy] = Group.value();
      const std::string ParamName = Arg.hasName() ?
                                      Arg.getName().str() :
                                      std::string("param_")
                                        + std::to_string(Group.index());
      IdentifierInfo *ParmId = &Context.Idents.get(makeCIdentifier(ParamName));
      ParmVarDecl *P = ParmVarDecl::Create(Context,
                                           NewFDecl,
                                           {},
                                           {},
                                           ParmId,
                                           ArgQualTy,
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

void DeclCreator::createFunctionAndCalleesDecl(clang::ASTContext &Ctx,
                                               llvm::Function *TheF) {

  revng_assert(TheF);
  auto FTags = FunctionTags::TagsSet::from(TheF);
  revng_assert(FTags.contains(FunctionTags::Isolated));

  std::set<const llvm::Function *> Called = getDirectlyCalledFunctions(*TheF);

  // Create the forward declaration of all the functions called by TheF
  for (auto *F : Called) {
    // Avoid forward declaring TheF
    if (F == TheF)
      continue;

    // Create the type decl necessary for declaring F.
    createTypeDeclsForFunctionPrototype(Ctx, F);

    // Create the FunctionDecl for F
    clang::FunctionDecl *NewFDecl = createFunDecl(Ctx,
                                                  F,
                                                  /* IsDefinition */ false);
    GlobalDecls[F] = NewFDecl;
  }

  // This is actually a definition, because the isolated function will
  // be fully decompiled and it needs a body.
  // This definition starts as a declaration that is than inflated by the
  // ASTBuildAnalysis.
  clang::FunctionDecl *NewFDecl = createFunDecl(Ctx,
                                                TheF,
                                                /* IsDefinition */ true);
  GlobalDecls[TheF] = NewFDecl;
}
