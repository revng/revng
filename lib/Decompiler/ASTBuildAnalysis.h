#pragma once

/// \brief DataFlow analysis to build the AST for a Function

//
// Copyright rev.ng Srls. See LICENSE.md for details.
//

#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Instructions.h"

#include "clang/Basic/Specifiers.h"

#include "revng/ADT/SmallMap.h"
#include "revng/Support/Assert.h"
#include "revng/Support/MonotoneFramework.h"

namespace clang {
class ASTContext;
class Expr;
class FieldDecl;
class FunctionDecl;
class LabelDecl;
class VarDecl;
class Stmt;
class TypeDecl;
} // namespace clang

namespace llvm {
class AllocaInst;
class BasicBlock;
class Constant;
class Function;
class PHINode;
class Type;
} // namespace llvm

namespace IR2AST {

using AllocaVarDeclMap = std::map<llvm::AllocaInst *, clang::VarDecl *>;
using BBLabelsMap = std::map<llvm::BasicBlock *, clang::LabelDecl *>;
using DeclMap = std::map<llvm::Instruction *, clang::VarDecl *>;
using FunctionsMap = std::map<llvm::Function *, clang::FunctionDecl *>;
using GlobalsMap = std::map<const llvm::GlobalVariable *, clang::VarDecl *>;
using StmtMap = std::map<llvm::Instruction *, clang::Stmt *>;
using StmtMultiMap = std::map<llvm::Instruction *,
                              llvm::SmallVector<clang::Stmt *, 2>>;
using TypeDeclMap = std::map<const llvm::Type *, clang::TypeDecl *>;
using FieldDeclMap = std::map<clang::TypeDecl *,
                              llvm::SmallVector<clang::FieldDecl *, 8>>;

class StmtBuilder {

private:
  using PHIIncomingMap = SmallMap<llvm::PHINode *, unsigned, 4>;
  using BBPHIMap = SmallMap<llvm::BasicBlock *, PHIIncomingMap, 4>;

  const std::set<llvm::Instruction *> &ToSerialize;
  clang::ASTContext &ASTCtx;
  uint64_t NVar;

public:
  AllocaVarDeclMap AllocaDecls;
  BBLabelsMap BBLabelDecls;
  DeclMap VarDecls;
  FunctionsMap &FunctionDecls;
  GlobalsMap &GlobalDecls;
  StmtMap InstrStmts;
  StmtMultiMap AdditionalStmts;
  TypeDeclMap TypeDecls;
  FieldDeclMap FieldDecls;
  BBPHIMap &BlockToPHIIncoming;

public:
  StmtBuilder(const std::set<llvm::Instruction *> &ToSerialize,
              clang::ASTContext &Ctx,
              GlobalsMap &GMap,
              FunctionsMap &FMap,
              BBPHIMap &BlockToPHIIncoming,
              TypeDeclMap &TypeDecls,
              FieldDeclMap &FieldDecls) :
    ToSerialize(ToSerialize),
    ASTCtx(Ctx),
    NVar(0),
    AllocaDecls(),
    BBLabelDecls(),
    VarDecls(),
    FunctionDecls(FMap),
    GlobalDecls(GMap),
    InstrStmts(),
    TypeDecls(TypeDecls),
    FieldDecls(FieldDecls),
    BlockToPHIIncoming(BlockToPHIIncoming) {}

  void createAST(llvm::Function &F, clang::FunctionDecl &FD);

  clang::Expr *getExprForValue(llvm::Value *V);
  clang::Expr *getUIntLiteral(uint64_t U);
  clang::Expr *getBoolLiteral(bool V);
  clang::Expr *getLiteralFromConstant(llvm::Constant *C);

  clang::VarDecl *getOrCreateLoopStateVarDecl(clang::FunctionDecl &FD);
  clang::VarDecl *getOrCreateSwitchStateVarDecl(clang::FunctionDecl &FD);

  clang::VarDecl *getLoopStateVarDecl() const { return LoopStateVarDecl; }
  clang::VarDecl *getSwitchStateVarDecl() const { return SwitchStateVarDecl; }

private:
  clang::VarDecl *createVarDecl(llvm::Instruction *I, clang::FunctionDecl &FD);
  clang::Stmt *buildStmt(llvm::Instruction &I);
  clang::Expr *createRValueExprForBinaryOperator(llvm::Instruction &I);
  clang::Expr *getParenthesizedExprForValue(llvm::Value *V);

private:
  clang::VarDecl *LoopStateVarDecl = nullptr;
  clang::VarDecl *SwitchStateVarDecl = nullptr;
};
} // namespace IR2AST
