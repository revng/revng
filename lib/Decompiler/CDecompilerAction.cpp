//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// LLVM includes
#include <llvm/IR/Constants.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/Type.h>

// clang includes
#include <clang/AST/Expr.h>
#include <clang/AST/Stmt.h>
#include <clang/Basic/SourceLocation.h>

// revng includes
#include <revng/Support/Assert.h>

// local libraries includes
#include "revng-c/RestructureCFGPass/ASTTree.h"
#include "revng-c/RestructureCFGPass/ExprNode.h"
#include "revng-c/RestructureCFGPass/RegionCFGTree.h"

// local includes
#include "ASTBuildAnalysis.h"
#include "CDecompilerBeautify.h"
#include "FuncDeclCreationAction.h"
#include "GlobalDeclCreationAction.h"
#include "IRASTTypeTranslation.h"
#include "MarkForSerialization.h"

#include "CDecompilerAction.h"

namespace clang {
namespace tooling {

using GlobalsMap = GlobalDeclCreationAction::GlobalsMap;
using FunctionsMap = FuncDeclCreationAction::FunctionsMap;

static void buildAndAppendSmts(SmallVectorImpl<clang::Stmt *> &Stmts,
                               ASTNode *N,
                               clang::ASTContext &ASTCtx,
                               IR2AST::StmtBuilder &ASTBuilder);

static clang::CompoundStmt *
buildCompoundScope(ASTNode *N,
                   clang::ASTContext &ASTCtx,
                   IR2AST::StmtBuilder &ASTBuilder,
                   SmallVector<clang::Stmt*, 32> AdditionalStmts = {}) {
  SmallVector<clang::Stmt *, 32> Stmts;
  buildAndAppendSmts(Stmts, N, ASTCtx, ASTBuilder);

  // Add additional statement to handle while e dowhile condition computation.
  Stmts.append(AdditionalStmts.begin(), AdditionalStmts.end());
  return CompoundStmt::Create(ASTCtx, Stmts, {}, {});
}

static clang::Expr *negateExpr(clang::ASTContext &ASTCtx, clang::Expr *E) {
  if (isa<clang::BinaryOperator>(E) or isa<clang::ConditionalOperator>(E))
    E = new (ASTCtx) ParenExpr({}, {}, E);
  using Unary = clang::UnaryOperator;
  E = new (ASTCtx) Unary(E,
                         UnaryOperatorKind::UO_Not,
                         E->getType(),
                         VK_RValue,
                         OK_Ordinary,
                         {},
                         false);
  return E;
}

static clang::Expr *createCondExpr(ExprNode *E,
                                   clang::ASTContext &ASTCtx,
                                   SmallVectorImpl<clang::Stmt *> &Stmts,
                                   IR2AST::StmtBuilder &ASTBuilder) {
  struct StackElement {
    ExprNode *Node;
    llvm::SmallVector<clang::Expr *, 2> ResolvedOperands;
  };
  llvm::SmallVector<StackElement, 4> VisitStack;
  clang::Expr *Result = nullptr;
  VisitStack.push_back( { nullptr, {} } );
  VisitStack.push_back( { E, {} });
  bool FoundBasicBlock = false;

  revng_assert(VisitStack.size() == 2);
  while (VisitStack.size() > 1) {
    StackElement &Current = VisitStack.back();
    switch (Current.Node->getKind()) {
    case ExprNode::NodeKind::NK_Atomic: {
      AtomicNode *Atomic = cast<AtomicNode>(Current.Node);
      llvm::BasicBlock *BB = Atomic->getConditionalBasicBlock();

      auto End = ASTBuilder.InstrStmts.end();
      for (llvm::Instruction &Instr : *BB) {
        auto It = ASTBuilder.InstrStmts.find(&Instr);
        if (It != End) {
          revng_assert(not FoundBasicBlock);
          Stmts.push_back(It->second);
        }
      }
      FoundBasicBlock = true;

      llvm::Instruction *CondTerminator = BB->getTerminator();
      llvm::BranchInst *Br = cast<llvm::BranchInst>(CondTerminator);
      revng_assert(Br->isConditional());
      llvm::Value *CondValue = Br->getCondition();
      clang::Expr *CondExpr = ASTBuilder.getExprForValue(CondValue);
      VisitStack.pop_back();
      VisitStack.back().ResolvedOperands.push_back(CondExpr);
    } break;
    case ExprNode::NodeKind::NK_Not: {
      NotNode *N = cast<NotNode>(Current.Node);
      revng_assert(Current.ResolvedOperands.size() <= 1);
      if (Current.ResolvedOperands.size() != 1) {
        ExprNode *Negated = N->getNegatedNode();
        VisitStack.push_back( {Negated, {} } );
      } else {
        clang::Expr *NotExpr = negateExpr(ASTCtx, Current.ResolvedOperands[0]);
        VisitStack.pop_back();
        VisitStack.back().ResolvedOperands.push_back(NotExpr);
      }
    } break;
    case ExprNode::NodeKind::NK_And:
    case ExprNode::NodeKind::NK_Or: {
      unsigned NumOperands = Current.ResolvedOperands.size();
      revng_assert(NumOperands <= 1);
      using ExprPair = std::pair<ExprNode *, ExprNode *>;
      BinaryNode *Binary = cast<BinaryNode>(Current.Node);
      if (NumOperands != 2) {
        ExprPair Childs = Binary->getInternalNodes();
        ExprNode *Op = (NumOperands == 0) ? Childs.first : Childs.second;
        VisitStack.push_back( {Op, {} } );
      } else {
        BinaryOperatorKind BinOpKind = isa<AndNode>(Binary) ?
                                       clang::BinaryOperatorKind::BO_And :
                                       clang::BinaryOperatorKind::BO_Or;
        clang::Expr *LHS = Current.ResolvedOperands[0];
        clang::Expr *RHS = Current.ResolvedOperands[1];
        clang::Expr *BinExpr = new (ASTCtx)
                               clang::BinaryOperator(LHS,
                                                     RHS,
                                                     BinOpKind,
                                                     LHS->getType(),
                                                     VK_RValue,
                                                     OK_Ordinary,
                                                     {},
                                                     FPOptions());
        VisitStack.pop_back();
        VisitStack.back().ResolvedOperands.push_back(BinExpr);
      }
    } break;
    default:
      revng_abort();
    }
  }
  revng_assert(VisitStack.size() == 1);
  revng_assert(VisitStack.back().ResolvedOperands.size() == 1);
  return VisitStack.back().ResolvedOperands[0];
}

static void buildAndAppendSmts(SmallVectorImpl<clang::Stmt *> &Stmts,
                               ASTNode *N,
                               clang::ASTContext &ASTCtx,
                               IR2AST::StmtBuilder &ASTBuilder) {

  if (N == nullptr)
    return;
  switch (N->getKind()) {
  case ASTNode::NodeKind::NK_Break:
    Stmts.push_back(new (ASTCtx) clang::BreakStmt(SourceLocation{}));
    break;
  case ASTNode::NodeKind::NK_Continue: {
    ContinueNode *Continue = cast<ContinueNode>(N);

    // Print the condition computation code of the if statement.
    if (Continue->hasComputation()) {
      IfNode *ComputationIfNode = Continue->getComputationIfNode();
      createCondExpr(ComputationIfNode->getCondExpr(), ASTCtx, Stmts, ASTBuilder);
    }
    Stmts.push_back(new (ASTCtx) clang::ContinueStmt(SourceLocation{}));
    break;
  }
  case ASTNode::NodeKind::NK_Code: {
    CodeNode *Code = cast<CodeNode>(N);
    llvm::BasicBlock *BB = Code->getOriginalBB();
    revng_assert(BB != nullptr);
    auto End = ASTBuilder.InstrStmts.end();
    for (llvm::Instruction &Instr : *BB) {
      auto It = ASTBuilder.InstrStmts.find(&Instr);
      if (It != End)
        Stmts.push_back(It->second);
    }
    break;
  }
  case ASTNode::NodeKind::NK_If: {
    IfNode *If = cast<IfNode>(N);
    clang::Expr *CondExpr = createCondExpr(If->getCondExpr(), ASTCtx, Stmts, ASTBuilder);
    revng_assert(CondExpr != nullptr);
    clang::Stmt *ThenScope = buildCompoundScope(If->getThen(), ASTCtx, ASTBuilder);
    clang::Stmt *ElseScope = buildCompoundScope(If->getElse(), ASTCtx, ASTBuilder);
    Stmts.push_back(new (ASTCtx) IfStmt(ASTCtx,
                                        {},
                                        false,
                                        nullptr,
                                        nullptr,
                                        CondExpr,
                                        ThenScope,
                                        {},
                                        ElseScope));
    break;
  }
  case ASTNode::NodeKind::NK_Scs: {
    ScsNode *LoopBody = cast<ScsNode>(N);

    if (LoopBody->isDoWhile()) {
      SmallVector<clang::Stmt*, 32> AdditionalStmts;

      // This shold retrieve the if which generates the condition of the loop
      // by accesing a dedicated field in the ScsNode.
      IfNode *LoopCondition = LoopBody->getRelatedCondition();
      clang::Expr *CondExpr = createCondExpr(LoopCondition->getCondExpr(),
                                             ASTCtx, AdditionalStmts, ASTBuilder);

      clang::Stmt *Body = buildCompoundScope(LoopBody->getBody(), ASTCtx,
                                             ASTBuilder, AdditionalStmts);

      for (clang::Stmt * S : AdditionalStmts)
        Stmts.push_back(S);
      Stmts.push_back(new (ASTCtx) DoStmt(Body, CondExpr, {}, {}, {}));
    } else if (LoopBody->isWhile()) {

      // This shold retrieve the if which generates the condition of the loop
      // by accesing a dedicated field in the ScsNode.
      IfNode *LoopCondition = LoopBody->getRelatedCondition();
      clang::Expr *CondExpr = createCondExpr(LoopCondition->getCondExpr(),
                                             ASTCtx, Stmts, ASTBuilder);
      clang::Stmt *Body = buildCompoundScope(LoopBody->getBody(), ASTCtx,
                                             ASTBuilder, {});
      Stmts.push_back(new (ASTCtx)
                        WhileStmt(ASTCtx, nullptr, CondExpr, Body, {}));
    } else {

      // Standard case.
      clang::Stmt *Body = buildCompoundScope(LoopBody->getBody(),
                                             ASTCtx, ASTBuilder);
      QualType UInt = ASTCtx.UnsignedIntTy;
      uint64_t UIntSize = ASTCtx.getTypeSize(UInt);
      clang::Expr *TrueCond = IntegerLiteral::Create(ASTCtx,
                                                     llvm::APInt(UIntSize, 1),
                                                     UInt,
                                                     {});

      Stmts.push_back(new (ASTCtx)
                      WhileStmt(ASTCtx, nullptr, TrueCond, Body, {}));
    }
    break;
  }
  case ASTNode::NodeKind::NK_List: {
    SequenceNode *Seq = cast<SequenceNode>(N);
    for (ASTNode *Child : Seq->nodes())
      buildAndAppendSmts(Stmts, Child, ASTCtx, ASTBuilder);
    break;
  }
  case ASTNode::NodeKind::NK_Switch: {
    SwitchNode *Switch = cast<SwitchNode>(N);
    llvm::Value *CondVal = Switch->getCondition();
    clang::Expr *CondExpr = ASTBuilder.getExprForValue(CondVal);
    clang::SwitchStmt *SwitchStatement = new (ASTCtx) SwitchStmt(ASTCtx,
                                                                 nullptr,
                                                                 nullptr,
                                                                 CondExpr);
    SmallVector<clang::Stmt *, 8> BodyStmts;
    for (auto &Pair : Switch->cases()) {
      llvm::ConstantInt *CaseVal = Pair.first;
      ASTNode *CaseNode = Pair.second;
      clang::Expr *CaseExpr = ASTBuilder.getExprForValue(CaseVal);
      clang::CaseStmt *Case = new (ASTCtx) CaseStmt(CaseExpr, nullptr,
                                                    {}, {}, {});
      clang::Stmt *CaseBody = buildCompoundScope(CaseNode, ASTCtx, ASTBuilder);
      Case->setSubStmt(CaseBody);
      BodyStmts.push_back(Case);
      BodyStmts.push_back(new (ASTCtx) clang::BreakStmt(SourceLocation{}));
      SwitchStatement->addSwitchCase(Case);
    }
    clang::Stmt *SwitchBody = CompoundStmt::Create(ASTCtx, BodyStmts, {}, {});
    SwitchStatement->setBody(SwitchBody);
    Stmts.push_back(SwitchStatement);
  } break;
  case ASTNode::NodeKind::NK_Set:
  case ASTNode::NodeKind::NK_IfCheck:
  default:
    revng_abort();
  }
}

static void buildFunctionBody(FunctionsMap::value_type &FPair,
                              ASTTree &CombedAST,
                              IR2AST::StmtBuilder &ASTBuilder) {
  llvm::Function &F = *FPair.first;
  clang::FunctionDecl *FDecl = FPair.second;
  ASTContext &ASTCtx = FDecl->getASTContext();
  SmallVector<clang::Decl *, 16> LocalVarDecls;
  for (auto &DeclPair : ASTBuilder.AllocaDecls)
    LocalVarDecls.push_back(DeclPair.second);
  for (auto &DeclPair : ASTBuilder.VarDecls)
    LocalVarDecls.push_back(DeclPair.second);

  SmallVector<clang::Stmt *, 32> BodyStmts;
  buildAndAppendSmts(BodyStmts, CombedAST.getRoot(), ASTCtx, ASTBuilder);

  unsigned NumLocalVars = LocalVarDecls.size();
  unsigned NumStmtsInBody = BodyStmts.size() + NumLocalVars;
  CompoundStmt *Body = CompoundStmt::CreateEmpty(ASTCtx, NumStmtsInBody);
  FDecl->setBody(Body);

  for (unsigned I = 0; I < NumLocalVars; ++I) {
    Decl *VDecl = LocalVarDecls[I];
    auto *LocalVarDeclStmt = new (ASTCtx) DeclStmt(DeclGroupRef(VDecl), {}, {});
    Body->body_begin()[I] = LocalVarDeclStmt;
  }

  for (unsigned I = NumLocalVars; I < NumStmtsInBody; ++I)
    Body->body_begin()[I] = BodyStmts[I - NumLocalVars];

#if 0
  int I = NumLocalVars;
  auto End = ASTInfo.InstrStmts.end();
  for (llvm::BasicBlock &BB : F) {
    SmallVector<clang::Stmt *, 16> BBStmts;
    for (llvm::Instruction &Instr : BB) {
      auto It = ASTInfo.InstrStmts.find(&Instr);
      if (It != End)
        BBStmts.push_back(It->second);
    }
    auto *BBCompoundStmt = CompoundStmt::Create(ASTCtx, BBStmts, {}, {});
    Body->body_begin()[I] = new (ASTCtx)
      LabelStmt({}, ASTInfo.LabelDecls.at(&BB), BBCompoundStmt);
    ++I;
  }
#endif
}

class Decompiler : public ASTConsumer {

private:

  using PHIIncomingMap = SmallMap<llvm::PHINode *, unsigned, 4>;
  using BBPHIMap = SmallMap<llvm::BasicBlock *, PHIIncomingMap, 4>;

public:
  explicit Decompiler(llvm::Function &F,
                      RegionCFG &RCFG,
                      ASTTree &CombedAST,
                      BBPHIMap &BlockToPHIIncoming,
                      std::unique_ptr<llvm::raw_ostream> Out) :
    TheF(F),
    RCFG(RCFG),
    CombedAST(CombedAST),
    BlockToPHIIncoming(BlockToPHIIncoming),
    Out(std::move(Out)) {}

  virtual void HandleTranslationUnit(ASTContext &Context) override {

    MarkForSerialization::Analysis Mark(TheF, RCFG);
    Mark.initialize();
    Mark.run();

    beautifyAST(TheF, CombedAST, Mark);

    using ConsumerPtr = std::unique_ptr<ASTConsumer>;
    FunctionsMap FunctionDecls;
    GlobalsMap GlobalVarAST;
    {
      // TODO: probably we don't need to use ASTConsumers for
      // CreateGlobalDeclCreator and CreateFuncDeclCreator. It's probably enough
      // to have standalone functions instead of full-fledged classes.

      // Build declaration of global variables
      ConsumerPtr GlobalDecls = CreateGlobalDeclCreator(TheF, GlobalVarAST);
      GlobalDecls->HandleTranslationUnit(Context);
      // Build function declaration
      ConsumerPtr FunDecls = CreateFuncDeclCreator(TheF, FunctionDecls);
      FunDecls->HandleTranslationUnit(Context);
    }

    revng_assert(not TheF.isDeclaration());
    revng_assert(TheF.getName().startswith("bb."));
    auto It = FunctionDecls.find(&TheF);
    revng_assert(It != FunctionDecls.end());
    clang::FunctionDecl *FunctionDecl = It->second;

    IR2AST::StmtBuilder ASTBuilder(TheF, Mark.getToSerialize(), Context,
                                   *FunctionDecl, GlobalVarAST, FunctionDecls,
                                   BlockToPHIIncoming);
    ASTBuilder.createAST();
    // auto &&ASTInfo = IR2ASTBuildAnalysis.extractASTInfo();

    buildFunctionBody(*It, CombedAST, ASTBuilder);

    // ConsumerPtr Dumper = CreateASTDumper(nullptr, "", true, false, false);
    // Dumper->HandleTranslationUnit(Context);
    ConsumerPtr Printer = CreateASTPrinter(std::move(Out), "");
    Printer->HandleTranslationUnit(Context);
  }

private:
  llvm::Function &TheF;
  RegionCFG &RCFG;
  ASTTree &CombedAST;
  std::unique_ptr<llvm::raw_ostream> Out;
  BBPHIMap &BlockToPHIIncoming;
};

std::unique_ptr<ASTConsumer> CDecompilerAction::newASTConsumer() {
  return std::make_unique<Decompiler>(F, RCFG, CombedAST,
                                      BlockToPHIIncoming, std::move(O));
}

} // end namespace tooling
} // end namespace clang
