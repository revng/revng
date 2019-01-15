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
class Function;
class PHINode;
} // namespace llvm

namespace IR2AST {

using PHIIncomingMap = SmallMap<const llvm::PHINode *, unsigned, 8>;
using BlockToPHIIncomingMap = SmallMap<llvm::BasicBlock *, PHIIncomingMap, 8>;
using DeclMap = std::map<llvm::Instruction *, clang::VarDecl *>;
using StmtMap = std::map<llvm::Instruction *, clang::Stmt *>;

struct SerializationInfo {
  std::map<llvm::AllocaInst *, clang::VarDecl *> AllocaDecls;
  std::map<llvm::BasicBlock *, clang::LabelDecl *> LabelDecls;
  DeclMap VarDecls;
  StmtMap InstrStmts;
  BlockToPHIIncomingMap BlockToPHIIncoming;
  std::map<llvm::Instruction *, clang::Stmt *> PendingExprs;
};

static_assert(std::is_move_constructible<SerializationInfo>::value,
              "SerializationInfo should be move constructible");

using LatticeElement = IntersectionMonotoneSet<llvm::Instruction *>;

class Analysis
  : public MonotoneFramework<Analysis,
                             llvm::BasicBlock *,
                             LatticeElement,
                             VisitType::ReversePostOrder,
                             llvm::SmallVector<llvm::BasicBlock *, 2>> {
private:
  llvm::Function &F;
  clang::FunctionDecl &FDecl;
  clang::ASTContext &ASTCtx;

  using GlobalsMap = std::map<const llvm::GlobalVariable *, clang::VarDecl *>;
  GlobalsMap &GlobalVarAST;
  using FunctionsMap = std::map<llvm::Function *, clang::FunctionDecl *>;
  FunctionsMap &FunctionAST;

  uint64_t NVar;
  SerializationInfo ASTInfo;

public:
  using Base = MonotoneFramework<Analysis,
                                 llvm::BasicBlock *,
                                 LatticeElement,
                                 VisitType::ReversePostOrder,
                                 llvm::SmallVector<llvm::BasicBlock *, 2>>;

  void assertLowerThanOrEqual(const LatticeElement &A,
                              const LatticeElement &B) const {
    revng_abort();
  }

  Analysis(llvm::Function &F,
           clang::ASTContext &Ctx,
           clang::FunctionDecl &FD,
           GlobalsMap &GMap,
           FunctionsMap &FMap) :
    Base(&F.getEntryBlock()),
    F(F),
    FDecl(FD),
    ASTCtx(Ctx),
    GlobalVarAST(GMap),
    FunctionAST(FMap) {
    Base::registerExtremal(&F.getEntryBlock());
  }

  void dumpFinalState() const { revng_abort(); }

  SerializationInfo &&extractASTInfo() { return std::move(ASTInfo); }

  llvm::SmallVector<llvm::BasicBlock *, 2>
  successors(llvm::BasicBlock *BB, InterruptType &) const {
    llvm::SmallVector<llvm::BasicBlock *, 2> Result;
    for (llvm::BasicBlock *Successor : make_range(succ_begin(BB), succ_end(BB)))
      Result.push_back(Successor);
    return Result;
  }

  llvm::Optional<LatticeElement>
  handleEdge(const LatticeElement & /*Original*/,
             llvm::BasicBlock * /*Source*/,
             llvm::BasicBlock * /*Destination*/) const {
    return llvm::Optional<LatticeElement>();
  }

  size_t successor_size(llvm::BasicBlock *BB, InterruptType &) const {
    return succ_end(BB) - succ_begin(BB);
  }

  static LatticeElement extremalValue(llvm::BasicBlock *) {
    return LatticeElement::top();
  }

  InterruptType transfer(llvm::BasicBlock *);

  void initialize();

private:
  void initInternal();
  void computePHIVars();

  void markValueToSerialize(llvm::Instruction *I);
  void markSetToSerialize(const LatticeElement &S) {
    for (llvm::Instruction *I: S)
      markValueToSerialize(I);
  }

  DeclMap::iterator createVarDecl(llvm::Instruction *I);
  void createVarDecls(const LatticeElement &S) {
    for (llvm::Instruction *I : S) {
      revng_assert(ASTInfo.VarDecls.count(I) == 0);
      createVarDecl(I);
    }
  }

  clang::Stmt *buildAST(llvm::Instruction &I);

  clang::Expr *getExprForValue(llvm::Value *V);
  clang::Expr *getParenthesizedExprForValue(llvm::Value *V);
  clang::Expr *createRValueExprForBinaryOperator(llvm::Instruction &I);
  clang::Expr *getLiteralFromConstant(llvm::Constant *C);
};

} // namespace IR2AST
#endif // REVNGC_ASTBUILDANALYSIS_H
