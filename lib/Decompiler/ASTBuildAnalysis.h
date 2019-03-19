#ifndef REVNGC_ASTBUILDANALYSIS_H
#define REVNGC_ASTBUILDANALYSIS_H
/// \brief DataFlow analysis to build the AST for a Function

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// LLVM includes
#include <llvm/ADT/SmallVector.h>
#include <llvm/IR/Dominators.h>
#include <llvm/IR/Instructions.h>

// clang includes
#include <clang/Basic/Specifiers.h>

// revng includes
#include <revng/ADT/SmallMap.h>
#include <revng/Support/Assert.h>
#include <revng/Support/MonotoneFramework.h>

namespace clang {
class FunctionDecl;
class LabelDecl;
class VarDecl;
class Stmt;
} // namespace clang

namespace llvm {
class AllocaInst;
class BasicBlock;
class Constant;
class Function;
class PHINode;
} // namespace llvm

namespace IR2AST {

using AllocaVarDeclMap = std::map<llvm::AllocaInst *, clang::VarDecl *>;
using BBLabelsMap = std::map<llvm::BasicBlock *, clang::LabelDecl *>;
using DeclMap = std::map<llvm::Instruction *, clang::VarDecl *>;
using FunctionsMap = std::map<llvm::Function *, clang::FunctionDecl *>;
using GlobalsMap = std::map<const llvm::GlobalVariable *, clang::VarDecl *>;
using StmtMap = std::map<llvm::Instruction *, clang::Stmt *>;

using LatticeElement = IntersectionMonotoneSet<llvm::Instruction *>;

class StmtBuilder {

private:

  using PHIIncomingMap = SmallMap<llvm::PHINode *, unsigned, 4>;
  using BBPHIMap = SmallMap<llvm::BasicBlock *, PHIIncomingMap, 4>;

  llvm::Function &F;
  const std::set<llvm::Instruction *> &ToSerialize;
  clang::FunctionDecl &FDecl;
  clang::ASTContext &ASTCtx;
  uint64_t NVar;

public:

  AllocaVarDeclMap AllocaDecls;
  BBLabelsMap BBLabelDecls;
  DeclMap VarDecls;
  FunctionsMap &FunctionDecls;
  GlobalsMap &GlobalDecls;
  StmtMap InstrStmts;
  BBPHIMap &BlockToPHIIncoming;

public:

  StmtBuilder(llvm::Function &F,
              const std::set<llvm::Instruction *> &ToSerialize,
              clang::ASTContext &Ctx,
              clang::FunctionDecl &FD,
              GlobalsMap &GMap,
              FunctionsMap &FMap,
              BBPHIMap &BlockToPHIIncoming) :
    F(F),
    ToSerialize(ToSerialize),
    FDecl(FD),
    ASTCtx(Ctx),
    NVar(0),
    AllocaDecls(),
    BBLabelDecls(),
    VarDecls(),
    FunctionDecls(FMap),
    GlobalDecls(GMap),
    InstrStmts(),
    BlockToPHIIncoming(BlockToPHIIncoming) {}

  // SerializationInfo &&extractASTInfo() { return std::move(ASTInfo); }

  void createAST();

  clang::Expr *getExprForValue(llvm::Value *V);
private:
  clang::VarDecl *createVarDecl(llvm::Instruction *I);
  clang::Stmt *buildStmt(llvm::Instruction &I);
  clang::Expr *createRValueExprForBinaryOperator(llvm::Instruction &I);
  clang::Expr *getParenthesizedExprForValue(llvm::Value *V);
  clang::Expr *getLiteralFromConstant(llvm::Constant *C);
};
} // namespace IR2AST
#endif // REVNGC_ASTBUILDANALYSIS_H
