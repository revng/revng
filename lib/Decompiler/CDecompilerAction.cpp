//
// Copyright rev.ng Srls. See LICENSE.md for details.
//

#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/Type.h"

#include "clang/AST/Decl.h"
#include "clang/AST/Expr.h"
#include "clang/AST/Stmt.h"
#include "clang/Basic/SourceLocation.h"

#include "revng/ADT/RecursiveCoroutine.h"
#include "revng/Support/Assert.h"
#include "revng/Support/Debug.h"
#include "revng/Support/FunctionTags.h"

#include "revng-c/RestructureCFGPass/ASTTree.h"
#include "revng-c/RestructureCFGPass/ExprNode.h"
#include "revng-c/RestructureCFGPass/RegionCFGTree.h"

#include "ASTBuildAnalysis.h"
#include "CDecompilerAction.h"
#include "DecompilationHelpers.h"
#include "IRASTTypeTranslation.h"

namespace clang {
namespace tooling {

using PHIIncomingMap = SmallMap<llvm::PHINode *, unsigned, 4>;

static void buildAndAppendSmts(clang::FunctionDecl &FDecl,
                               SmallVectorImpl<clang::Stmt *> &Stmts,
                               ASTNode *N,
                               clang::ASTContext &ASTCtx,
                               IR2AST::StmtBuilder &ASTBuilder,
                               const SerializationMap &Mark);

static clang::CompoundStmt *
buildCompoundScope(clang::FunctionDecl &FDecl,
                   ASTNode *N,
                   clang::ASTContext &ASTCtx,
                   IR2AST::StmtBuilder &ASTBuilder,
                   const SerializationMap &Mark,
                   SmallVector<clang::Stmt *, 32> AdditionalStmts = {}) {
  SmallVector<clang::Stmt *, 32> Stmts;
  buildAndAppendSmts(FDecl, Stmts, N, ASTCtx, ASTBuilder, Mark);

  // Add additional statement to handle while e dowhile condition computation.
  Stmts.append(AdditionalStmts.begin(), AdditionalStmts.end());
  return CompoundStmt::Create(ASTCtx, Stmts, {}, {});
}

static clang::Expr *negateExpr(clang::ASTContext &ASTCtx, clang::Expr *E) {
  if (auto *BinOp = dyn_cast<clang::BinaryOperator>(E->IgnoreParenImpCasts())) {
    auto OpCode = BinOp->getOpcode();
    if (clang::BinaryOperator::isComparisonOp(OpCode)) {
      // For binary comparison operators, we can just invert the comparison and
      // return.
      clang::Expr *LHS = BinOp->getLHS();
      clang::Expr *RHS = BinOp->getRHS();
      auto NegatedOpCode = clang::BinaryOperator::negateComparisonOp(OpCode);
      clang::Expr *Res = clang::BinaryOperator::Create(ASTCtx,
                                                       LHS,
                                                       RHS,
                                                       NegatedOpCode,
                                                       BinOp->getType(),
                                                       VK_RValue,
                                                       OK_Ordinary,
                                                       {},
                                                       FPOptions());
      return Res;
    }
  }

  if (isa<clang::BinaryOperator>(E) or isa<clang::ConditionalOperator>(E))
    E = new (ASTCtx) ParenExpr({}, {}, E);

  QualType ExprTy = E->getType();
  bool IsBool = ExprTy.getTypePtr()->isBooleanType();
  auto OpCode = IsBool ? UnaryOperatorKind::UO_LNot : UnaryOperatorKind::UO_Not;
  return clang::UnaryOperator::Create(ASTCtx,
                                      E,
                                      OpCode,
                                      ExprTy,
                                      VK_RValue,
                                      OK_Ordinary,
                                      {},
                                      false,
                                      FPOptions());
}

static void buildStmtsForBasicBlock(llvm::BasicBlock *BB,
                                    clang::ASTContext &ASTCtx,
                                    SmallVectorImpl<clang::Stmt *> &Stmts,
                                    IR2AST::StmtBuilder &ASTBuilder,
                                    const SerializationMap &Mark) {
  revng_assert(BB != nullptr);
  auto StmtEnd = ASTBuilder.InstrStmts.end();
  auto VDeclEnd = ASTBuilder.VarDecls.end();
  auto AdditionalStmtsEnd = ASTBuilder.AdditionalStmts.end();
  for (llvm::Instruction &Instr : *BB) {

    // Skip llvm.assume() instrinsics
    if (auto *Call = dyn_cast<llvm::CallInst>(&Instr))
      if (Call->getIntrinsicID() == llvm::Intrinsic::assume)
        continue;

    // Skip instructions that do not need to be serialized.
    if (auto MarkIt = Mark.find(&Instr);
        MarkIt == Mark.end()
        or not SerializationFlags::mustBeSerialized(MarkIt->second))
      continue;

    auto StmtIt = ASTBuilder.InstrStmts.find(&Instr);
    if (StmtIt != StmtEnd and StmtIt->second != nullptr) {
      clang::Stmt *EmittedStmt = nullptr;
      auto VarDeclIt = ASTBuilder.VarDecls.find(&Instr);
      if (VarDeclIt != VDeclEnd) {
        clang::VarDecl *VDecl = VarDeclIt->second;
        QualType VarType = VDecl->getType();
        clang::Expr *LHS = new (ASTCtx)
          DeclRefExpr(ASTCtx, VDecl, false, VarType, VK_LValue, {});
        clang::Expr *RHS = cast<clang::Expr>(StmtIt->second);
        if (RHS->getType() != VarType) {
          if (isa<clang::BinaryOperator>(RHS))
            RHS = new (ASTCtx) ParenExpr({}, {}, RHS);
          RHS = createCast(VarType, RHS, ASTCtx);
        }
        EmittedStmt = clang::BinaryOperator::Create(ASTCtx,
                                                    LHS,
                                                    RHS,
                                                    BO_Assign,
                                                    VarType,
                                                    VK_RValue,
                                                    OK_Ordinary,
                                                    {},
                                                    FPOptions());
      } else {
        EmittedStmt = StmtIt->second;
      }
      Stmts.push_back(EmittedStmt);
    }
    auto AdditionalStmtsIt = ASTBuilder.AdditionalStmts.find(&Instr);
    if (AdditionalStmtsIt != AdditionalStmtsEnd)
      for (clang::Stmt *S : AdditionalStmtsIt->second)
        Stmts.push_back(S);
  }

  // Print assignments of PHI variables where needed
  auto PHIMapIt = ASTBuilder.BlockToPHIIncoming.find(BB);
  if (PHIMapIt != ASTBuilder.BlockToPHIIncoming.end()) {
    using Pair = PHIIncomingMap::value_type;
    for (Pair &P : PHIMapIt->second) {
      llvm::PHINode *ThePHI = P.first;
      unsigned IncomingIdx = P.second;
      revng_assert(ThePHI != nullptr);

      clang::VarDecl *PHIVarDecl = ASTBuilder.VarDecls.at(ThePHI);
      QualType VarType = PHIVarDecl->getType();
      clang::Expr *LHS = new (ASTCtx)
        DeclRefExpr(ASTCtx, PHIVarDecl, false, VarType, VK_LValue, {});

      llvm::Value *IncomingV = ThePHI->getIncomingValue(IncomingIdx);
      clang::Expr *RHS = ASTBuilder.getExprForValue(IncomingV);
      if (RHS->getType() != VarType) {
        if (isa<clang::BinaryOperator>(RHS))
          RHS = new (ASTCtx) ParenExpr({}, {}, RHS);
        RHS = createCast(VarType, RHS, ASTCtx);
      }
      clang::Stmt *EmittedStmt = nullptr;
      EmittedStmt = clang::BinaryOperator::Create(ASTCtx,
                                                  LHS,
                                                  RHS,
                                                  BO_Assign,
                                                  VarType,
                                                  VK_RValue,
                                                  OK_Ordinary,
                                                  {},
                                                  FPOptions());
      Stmts.push_back(EmittedStmt);
    }
  }
}

static clang::Expr *createCondExpr(ExprNode *E,
                                   clang::ASTContext &ASTCtx,
                                   SmallVectorImpl<clang::Stmt *> &Stmts,
                                   IR2AST::StmtBuilder &ASTBuilder,
                                   const SerializationMap &Mark) {
  struct StackElement {
    ExprNode *Node;
    llvm::SmallVector<clang::Expr *, 2> ResolvedOperands;
  };
  llvm::SmallVector<StackElement, 4> VisitStack;
  VisitStack.push_back({ nullptr, {} });
  VisitStack.push_back({ E, {} });

  revng_assert(VisitStack.size() == 2);
  while (VisitStack.size() > 1) {
    StackElement &Current = VisitStack.back();
    switch (Current.Node->getKind()) {

    case ExprNode::NodeKind::NK_Atomic: {
      AtomicNode *Atomic = cast<AtomicNode>(Current.Node);
      llvm::BasicBlock *BB = Atomic->getConditionalBasicBlock();
      revng_assert(BB != nullptr);
      buildStmtsForBasicBlock(BB, ASTCtx, Stmts, ASTBuilder, Mark);
      llvm::Instruction *CondTerminator = BB->getTerminator();
      llvm::BranchInst *Br = cast<llvm::BranchInst>(CondTerminator);
      revng_assert(Br->isConditional());
      llvm::Value *CondValue = Br->getCondition();
      clang::Expr *CondExpr = ASTBuilder.getExprForValue(CondValue);
      revng_assert(CondExpr);
      VisitStack.pop_back();
      VisitStack.back().ResolvedOperands.push_back(CondExpr);
    } break;

    case ExprNode::NodeKind::NK_Not: {
      NotNode *N = cast<NotNode>(Current.Node);
      revng_assert(Current.ResolvedOperands.size() <= 1);
      if (Current.ResolvedOperands.size() != 1) {
        ExprNode *Negated = N->getNegatedNode();
        VisitStack.push_back({ Negated, {} });
      } else {
        clang::Expr *NotExpr = negateExpr(ASTCtx, Current.ResolvedOperands[0]);
        VisitStack.pop_back();
        VisitStack.back().ResolvedOperands.push_back(NotExpr);
      }
    } break;

    case ExprNode::NodeKind::NK_And:
    case ExprNode::NodeKind::NK_Or: {
      size_t NumOperands = Current.ResolvedOperands.size();
      revng_assert(NumOperands <= 2);
      using ExprPair = std::pair<ExprNode *, ExprNode *>;
      BinaryNode *Binary = cast<BinaryNode>(Current.Node);

      if (NumOperands != 2) {
        ExprPair Childs = Binary->getInternalNodes();
        ExprNode *Op = (NumOperands == 0) ? Childs.first : Childs.second;
        VisitStack.push_back({ Op, {} });
      } else {
        clang::Expr *LHS = Current.ResolvedOperands[0];
        clang::Expr *RHS = Current.ResolvedOperands[1];
        bool BothBool = LHS->getType().getTypePtr()->isBooleanType()
                        and RHS->getType().getTypePtr()->isBooleanType();
        auto BinOpKind = isa<AndNode>(Binary) ?
                           (BothBool ? clang::BinaryOperatorKind::BO_LAnd :
                                       clang::BinaryOperatorKind::BO_And) :
                           (BothBool ? clang::BinaryOperatorKind::BO_LOr :
                                       clang::BinaryOperatorKind::BO_Or);
        clang::Expr *BinExpr = clang::BinaryOperator::Create(ASTCtx,
                                                             LHS,
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
    }
  }
  revng_assert(VisitStack.size() == 1);
  revng_assert(VisitStack.back().ResolvedOperands.size() == 1);
  return VisitStack.back().ResolvedOperands[0];
}

static void buildAndAppendSmts(clang::FunctionDecl &FDecl,
                               SmallVectorImpl<clang::Stmt *> &Stmts,
                               ASTNode *N,
                               clang::ASTContext &ASTCtx,
                               IR2AST::StmtBuilder &ASTBuilder,
                               const SerializationMap &Mark) {
  if (N == nullptr)
    return;

  auto Kind = N->getKind();
  switch (Kind) {

  case ASTNode::NodeKind::NK_Break: {
    BreakNode *Break = llvm::cast<BreakNode>(N);
    if (Break->breaksFromWithinSwitch()) {
      auto *StateVarD = ASTBuilder.getOrCreateSwitchStateVarDecl(FDecl);
      QualType T = StateVarD->getType();
      clang::Expr *State = new (ASTCtx)
        DeclRefExpr(ASTCtx, StateVarD, false, T, VK_LValue, {});

      clang::Expr *TrueVal = ASTBuilder.getBoolLiteral(true);
      QualType BoolTy = TrueVal->getType();
      clang::Stmt *AssignStmt = clang::BinaryOperator::Create(ASTCtx,
                                                              State,
                                                              TrueVal,
                                                              BO_Assign,
                                                              BoolTy,
                                                              VK_RValue,
                                                              OK_Ordinary,
                                                              {},
                                                              FPOptions());
      Stmts.push_back(AssignStmt);
    }
  };
    [[fallthrough]];

  case ASTNode::NodeKind::NK_SwitchBreak:
    Stmts.push_back(new (ASTCtx) clang::BreakStmt(SourceLocation{}));
    break;

  case ASTNode::NodeKind::NK_Continue: {
    ContinueNode *Continue = cast<ContinueNode>(N);

    // Print the condition computation code of the if statement.
    if (Continue->hasComputation()) {
      IfNode *ComputationIfNode = Continue->getComputationIfNode();
      createCondExpr(ComputationIfNode->getCondExpr(),
                     ASTCtx,
                     Stmts,
                     ASTBuilder,
                     Mark);
    }

    // Actually print the continue statement only if the continue is not
    // implicit (i.e. it is not the last statement of the loop).
    if (not Continue->isImplicit()) {
      Stmts.push_back(new (ASTCtx) clang::ContinueStmt(SourceLocation{}));
    }
  } break;

  case ASTNode::NodeKind::NK_Code: {
    CodeNode *Code = cast<CodeNode>(N);
    llvm::BasicBlock *BB = Code->getOriginalBB();
    revng_assert(BB != nullptr);
    buildStmtsForBasicBlock(BB, ASTCtx, Stmts, ASTBuilder, Mark);
  } break;

  case ASTNode::NodeKind::NK_If: {
    IfNode *If = cast<IfNode>(N);

    clang::Expr *CondExpr = createCondExpr(If->getCondExpr(),
                                           ASTCtx,
                                           Stmts,
                                           ASTBuilder,
                                           Mark);
    revng_assert(CondExpr != nullptr);

    revng_assert(nullptr != If->getThen());
    clang::Stmt *ThenScope = buildCompoundScope(FDecl,
                                                If->getThen(),
                                                ASTCtx,
                                                ASTBuilder,
                                                Mark);
    revng_assert(nullptr != ThenScope);

    // Handle the situation in which we do have a nullptr in the place of the
    // else node of the if statement, which may result in a non empty
    // `ElseScope` and therefore an empty compound statement.
    if (If->getElse() == nullptr) {
      Stmts.push_back(IfStmt::Create(ASTCtx,
                                     {},
                                     false,
                                     nullptr,
                                     nullptr,
                                     CondExpr,
                                     {},
                                     {},
                                     ThenScope,
                                     {},
                                     nullptr));
    } else {

      clang::Stmt *ElseScope = buildCompoundScope(FDecl,
                                                  If->getElse(),
                                                  ASTCtx,
                                                  ASTBuilder,
                                                  Mark);
      revng_assert(nullptr != ElseScope);
      Stmts.push_back(IfStmt::Create(ASTCtx,
                                     {},
                                     false,
                                     nullptr,
                                     nullptr,
                                     CondExpr,
                                     {},
                                     {},
                                     ThenScope,
                                     {},
                                     ElseScope));
    }

    break;
  }

  case ASTNode::NodeKind::NK_Scs: {
    ScsNode *LoopBody = cast<ScsNode>(N);

    if (LoopBody->isDoWhile()) {
      SmallVector<clang::Stmt *, 32> AdditionalStmts;

      // Retrieve the if that generates the condition of the loop.
      // Then create the condition expression, while at the same time filling
      // the additional statements that need to be printed to compute that
      // condition.
      IfNode *LoopCondition = LoopBody->getRelatedCondition();
      clang::Expr *CondExpr = createCondExpr(LoopCondition->getCondExpr(),
                                             ASTCtx,
                                             AdditionalStmts,
                                             ASTBuilder,
                                             Mark);

      clang::Stmt *Body = buildCompoundScope(FDecl,
                                             LoopBody->getBody(),
                                             ASTCtx,
                                             ASTBuilder,
                                             Mark,
                                             AdditionalStmts);

      Stmts.push_back(new (ASTCtx) DoStmt(Body, CondExpr, {}, {}, {}));
    } else {

      clang::Expr *CondExpr = nullptr;
      if (LoopBody->isWhile()) {

        // This shold retrieve the if which generates the condition of the loop
        // by accesing a dedicated field in the ScsNode.
        IfNode *LoopCondition = LoopBody->getRelatedCondition();
        CondExpr = createCondExpr(LoopCondition->getCondExpr(),
                                  ASTCtx,
                                  Stmts,
                                  ASTBuilder,
                                  Mark);

      } else { // the loop is a while(1)
        QualType UInt = ASTCtx.UnsignedIntTy;
        unsigned UIntSize = static_cast<unsigned>(ASTCtx.getTypeSize(UInt));
        CondExpr = IntegerLiteral::Create(ASTCtx,
                                          llvm::APInt(UIntSize, 1),
                                          UInt,
                                          {});
      }

      clang::Stmt *Body = buildCompoundScope(FDecl,
                                             LoopBody->getBody(),
                                             ASTCtx,
                                             ASTBuilder,
                                             Mark);

      auto
        *While = WhileStmt::Create(ASTCtx, nullptr, CondExpr, Body, {}, {}, {});
      Stmts.push_back(While);
    }
  } break;

  case ASTNode::NodeKind::NK_List: {
    SequenceNode *Seq = cast<SequenceNode>(N);
    for (ASTNode *Child : Seq->nodes())
      buildAndAppendSmts(FDecl, Stmts, Child, ASTCtx, ASTBuilder, Mark);
  } break;

  case ASTNode::NodeKind::NK_Switch: {
    SwitchNode *Switch = cast<SwitchNode>(N);

    // Generate the condition of the switch.
    clang::Expr *CondExpr = nullptr;
    llvm::Value *SwitchVar = Switch->getCondition();
    if (SwitchVar) {
      // If the switch is not weaved we need to print the instructions in the
      // basic block before it.
      if (not Switch->isWeaved()) {
        llvm::BasicBlock *BB = Switch->getOriginalBB();
        revng_assert(BB != nullptr); // This is not a switch dispatcher.
        buildStmtsForBasicBlock(BB, ASTCtx, Stmts, ASTBuilder, Mark);
      }

      CondExpr = ASTBuilder.getExprForValue(SwitchVar);
    } else {
      revng_assert(Switch->getOriginalBB() == nullptr);
      // This is a dispatcher switch, check the loop state variable
      clang::VarDecl *StateVarD = ASTBuilder.getOrCreateLoopStateVarDecl(FDecl);
      QualType T = StateVarD->getType();
      CondExpr = new (ASTCtx)
        DeclRefExpr(ASTCtx, StateVarD, false, T, VK_LValue, {});
    }
    revng_assert(CondExpr != nullptr);

    // Generate the switch statement
    clang::SwitchStmt *SwitchStatement = SwitchStmt::Create(ASTCtx,
                                                            nullptr,
                                                            nullptr,
                                                            CondExpr,
                                                            {},
                                                            {});

    // Generate the body of the switch
    SmallVector<clang::Stmt *, 8> BodyStmts;
    for (const auto &[Labels, CaseNode] : Switch->cases()) {
      revng_assert(not Labels.empty()); // The Default is handled separately

      // Build the body of the case. We build it before iterating on the case
      // labels, because we may have more than one case label with the same
      // body, such as in:
      // switch (x) {
      //     case 0:
      //     case 1:
      //     case 2:
      //       return 5;
      // }
      // So, first we build here the compound statement representing the scope
      // with return 5;
      clang::Stmt *CaseBody = buildCompoundScope(FDecl,
                                                 CaseNode,
                                                 ASTCtx,
                                                 ASTBuilder,
                                                 Mark);

      // Now we iterate on the case labels and we build them as clang produces
      // them, i. e. in the following shape
      //
      //  |-SwitchStmt
      //  | `-DeclRefExpr 'int' 'x'
      //  `-CompoundStmt
      //    |-CaseStmt
      //    | |-ConstantExpr 'int'
      //    | | `-IntegerLiteral 'int' 0
      //    | `-CaseStmt
      //    |   |-ConstantExpr 'int'
      //    |   | `-IntegerLiteral 'int' 1
      //    |   `-CaseStmt
      //    |     |-ConstantExpr 'int'
      //    |     | `-IntegerLiteral 'int' 2
      //    |     `-ReturnStmt
      //    |       `-IntegerLiteral 'int' 5
      llvm::SmallVector<clang::CaseStmt *, 8> Cases;
      for (uint64_t CaseVal : Labels) {

        clang::Expr *CaseExpr = nullptr;
        if (SwitchVar) {
          llvm::Type *SwitchVarT = SwitchVar->getType();
          auto *IntType = cast<llvm::IntegerType>(SwitchVarT);
          auto *CaseConst = llvm::ConstantInt::get(IntType, CaseVal);
          CaseExpr = ASTBuilder.getExprForValue(CaseConst);
        } else {
          CaseExpr = ASTBuilder.getUIntLiteral(CaseVal);
        }

        revng_assert(CaseExpr != nullptr);
        // Build the case
        clang::CaseStmt *Case = CaseStmt::Create(ASTCtx,
                                                 CaseExpr,
                                                 nullptr,
                                                 {},
                                                 {},
                                                 {});
        Case->setSubStmt(CaseBody);
        Cases.push_back(Case);
        // Set CaseBody to point to the last added Case, because this Case will
        // be the body of the next CaseStmt.
        CaseBody = Case;
      }
      revng_assert(llvm::isa<clang::CaseStmt>(CaseBody));
      BodyStmts.push_back(CaseBody);
      BodyStmts.push_back(new (ASTCtx) clang::BreakStmt(SourceLocation{}));

      // Do it in reverse order, so that cases are inserted in the same order
      // that you can see them in the emitted code. Not sure if this is
      // necessary, but just want to avoid problems.
      for (clang::CaseStmt *Case : llvm::reverse(Cases))
        SwitchStatement->addSwitchCase(Case);
    }

    if (auto *Default = Switch->getDefault()) {
      // Build the case
      auto *Def = new (ASTCtx) clang::DefaultStmt({}, {}, nullptr);
      // Build the body of the case
      clang::Stmt *DefBody = buildCompoundScope(FDecl,
                                                Default,
                                                ASTCtx,
                                                ASTBuilder,
                                                Mark);
      Def->setSubStmt(DefBody);
      BodyStmts.push_back(Def);
      BodyStmts.push_back(new (ASTCtx) clang::BreakStmt(SourceLocation{}));
      SwitchStatement->addSwitchCase(Def);
    }

    clang::Stmt *SwitchBody = CompoundStmt::Create(ASTCtx, BodyStmts, {}, {});
    SwitchStatement->setBody(SwitchBody);

    // If the switch needs a loop break dispatcher, reset the associated state
    // variable before emitting the switch statement.
    if (Switch->needsLoopBreakDispatcher()) {
      auto *StateVarD = ASTBuilder.getOrCreateSwitchStateVarDecl(FDecl);
      QualType T = StateVarD->getType();
      clang::Expr *State = new (ASTCtx)
        DeclRefExpr(ASTCtx, StateVarD, false, T, VK_LValue, {});

      clang::Expr *FalseInit = ASTBuilder.getBoolLiteral(false);
      QualType BoolTy = FalseInit->getType();
      clang::Stmt *AssignStmt = clang::BinaryOperator::Create(ASTCtx,
                                                              State,
                                                              FalseInit,
                                                              BO_Assign,
                                                              BoolTy,
                                                              VK_RValue,
                                                              OK_Ordinary,
                                                              {},
                                                              FPOptions());
      Stmts.push_back(AssignStmt);
    }

    Stmts.push_back(SwitchStatement);

    // If the switch needs it, generate a dispatcher to handle break
    // instructions inside the switch that are trying to break direcly out of a
    // loop that contains the switch
    if (Switch->needsLoopBreakDispatcher()) {
      // Build the AST for
      // if (CondExpr)
      //   break;
      auto *StateVarD = ASTBuilder.getOrCreateSwitchStateVarDecl(FDecl);
      QualType T = StateVarD->getType();
      CondExpr = new (ASTCtx)
        DeclRefExpr(ASTCtx, StateVarD, false, T, VK_LValue, {});
      clang::BreakStmt *Break = new (ASTCtx) clang::BreakStmt(SourceLocation{});
      Stmts.push_back(IfStmt::Create(ASTCtx,
                                     {},
                                     false,
                                     nullptr,
                                     nullptr,
                                     CondExpr,
                                     {},
                                     {},
                                     Break,
                                     {},
                                     nullptr));
    }
  } break;

  case ASTNode::NodeKind::NK_Set: {
    SetNode *Set = cast<SetNode>(N);
    auto *StateVarDecl = ASTBuilder.getOrCreateLoopStateVarDecl(FDecl);
    QualType Type = StateVarDecl->getType();
    clang::DeclRefExpr *StateVar = new (ASTCtx)
      DeclRefExpr(ASTCtx, StateVarDecl, false, Type, VK_LValue, {});

    unsigned StateValue = Set->getStateVariableValue();
    clang::Expr *StateValueUInt = ASTBuilder.getUIntLiteral(StateValue);
    QualType UIntType = StateValueUInt->getType();
    clang::Stmt *AssignStmt = clang::BinaryOperator::Create(ASTCtx,
                                                            StateVar,
                                                            StateValueUInt,
                                                            BO_Assign,
                                                            UIntType,
                                                            VK_RValue,
                                                            OK_Ordinary,
                                                            {},
                                                            FPOptions());
    Stmts.push_back(AssignStmt);

  } break;
  }
}

static void buildFunctionBody(llvm::Function *F,
                              clang::FunctionDecl *FDecl,
                              ASTTree &CombedAST,
                              IR2AST::StmtBuilder &ASTBuilder,
                              const SerializationMap &Mark) {
  ASTContext &ASTCtx = FDecl->getASTContext();

  // Check that the function we are attempting to decompile is not a variadic
  // function
  revng_assert(not FDecl->isVariadic());

  SmallVector<clang::Stmt *, 32> BodyStmts;
  buildAndAppendSmts(*FDecl,
                     BodyStmts,
                     CombedAST.getRoot(),
                     ASTCtx,
                     ASTBuilder,
                     Mark);

  SmallVector<clang::Decl *, 16> LocalVarDecls;
  // Allocas always go at the beginning of the function body.
  for (auto &DeclPair : ASTBuilder.AllocaDecls)
    LocalVarDecls.push_back(DeclPair.second);

  // Other VAriable declarations are emitted int the entry block for now.
  // In the future, we should emit local variable declarations
  // as-late-as-possible, right before they are assigned.
  for (auto &DeclPair : ASTBuilder.VarDecls)
    LocalVarDecls.push_back(DeclPair.second);

  // If we have a loop state variable, declare it in the entry block.
  // For now all the loops share the same state variable.
  // In the future we might decide that we want a separate loop state variable
  // for each loop, but this is not strictly necessary.
  if (clang::VarDecl *V = ASTBuilder.getLoopStateVarDecl())
    LocalVarDecls.push_back(V);

  // If we have a switch state variable, declare it in the entry block
  // For now all the switches share the same state variable.
  // In the future we might decide that we want a separate switch state variable
  // for each switch, but this is not strictly necessary.
  if (clang::VarDecl *V = ASTBuilder.getSwitchStateVarDecl())
    LocalVarDecls.push_back(V);

  auto NumLocalVars = LocalVarDecls.size();
  auto NumStmtsInBody = BodyStmts.size() + NumLocalVars;
  revng_check(static_cast<unsigned>(NumLocalVars) == NumLocalVars,
              "error: the function you are trying to decompile is too big!");
  unsigned BodySize = static_cast<unsigned>(NumStmtsInBody);
  unsigned VarSize = static_cast<unsigned>(NumLocalVars);
  revng_check(static_cast<unsigned>(NumStmtsInBody) == NumStmtsInBody,
              "error: the function you are trying to decompile is too big!");
  CompoundStmt *Body = CompoundStmt::CreateEmpty(ASTCtx, BodySize);
  FDecl->setBody(Body);

  for (unsigned I = 0; I < VarSize; ++I) {
    Decl *VDecl = LocalVarDecls[I];
    auto *LocalVarDeclStmt = new (ASTCtx) DeclStmt(DeclGroupRef(VDecl), {}, {});
    Body->body_begin()[I] = LocalVarDeclStmt;
  }

  for (unsigned I = VarSize; I < BodySize; ++I)
    Body->body_begin()[I] = BodyStmts[I - VarSize];

  /*
   * This is a leftover from when we used to print all the labels of the
   * basicblocks and put gotos all over the places.
   * I left it here as a reference, for when we plan to revive part on the work
   * on the gotos.
   */
  /*
  int I = VarSize;
  auto End = ASTInfo.InstrStmts.end();
  for (llvm::BasicBlock &BB : &F) {
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
  */
}

class Decompiler : public ASTConsumer {

private:
  using BBPHIMap = SmallMap<llvm::BasicBlock *, PHIIncomingMap, 4>;
  using DuplicationMap = std::map<const llvm::BasicBlock *, size_t>;

public:
  explicit Decompiler(llvm::Function &F,
                      ASTTree &CombedAST,
                      BBPHIMap &BlockToPHIIncoming,
                      const dla::ValueLayoutMap *VL,
                      llvm::ScalarEvolution *SCEV,
                      const SerializationMap &M,
                      std::unique_ptr<llvm::raw_ostream> Out) :
    TheF(F),
    CombedAST(CombedAST),
    BlockToPHIIncoming(BlockToPHIIncoming),
    ValueLayouts(VL),
    SE(SCEV),
    Mark(M),
    Out(std::move(Out)) {}

  virtual void HandleTranslationUnit(ASTContext &Context) override;

private:
  llvm::Function &TheF;
  ASTTree &CombedAST;
  BBPHIMap &BlockToPHIIncoming;
  const dla::ValueLayoutMap *ValueLayouts;
  llvm::ScalarEvolution *SE;
  const SerializationMap &Mark;
  std::unique_ptr<llvm::raw_ostream> Out;
};

static Logger<> TypeDeclOrderLog("type-decl-order");

using RecordDeclSetVector = llvm::SmallSetVector<clang::RecordDecl *, 16>;

static RecursiveCoroutine<void>
addDeclsInOrder(clang::TypeDecl *TD,
                RecordDeclSetVector &OrderedDecls,
                RecordDeclSetVector &NeedForwardDecl) {

  LoggerIndent Indent(TypeDeclOrderLog);
  revng_log(TypeDeclOrderLog, TD->getNameAsString());

  auto *StructDecl = llvm::dyn_cast_or_null<clang::RecordDecl>(TD);
  if (not StructDecl) {
    revng_log(TypeDeclOrderLog, "not a record");
    rc_return;
  }

  if (OrderedDecls.count(StructDecl)) {
    revng_log(TypeDeclOrderLog, "already declared");
    rc_return;
  }

  {
    LoggerIndent FieldIndent(TypeDeclOrderLog);
    for (clang::FieldDecl *FldDecl : StructDecl->fields()) {
      clang::QualType FieldQualType = FldDecl->getType();
      const clang::Type *FieldType = FieldQualType.getTypePtr();

      // If we have a field which is a struct, recur
      if (auto *FldStructType = FieldType->getAs<clang::RecordType>()) {
        rc_recur addDeclsInOrder(FldStructType->getDecl(),
                                 OrderedDecls,
                                 NeedForwardDecl);
        continue;
      }

      // If we have a field which is a pointer type we have to look into it.
      auto *PtrTy = FieldType->getAs<clang::PointerType>();

      // But first handle array fields, namely arrays of structs, and arrays of
      // pointers.
      // We handle array of structs explicitly, and for arrays of pointers we
      // just fall back to the regular pointer handling.
      auto *ArrayTy = FieldType->getAsArrayTypeUnsafe();
      if (not PtrTy and ArrayTy) {
        // Peel off arbitrary nested arrays, until we get to the underlying
        // element type, that is not an array.
        const clang::Type *ElemTy = nullptr;
        {
          do {
            clang::QualType ElemQualTy = ArrayTy->getElementType();
            ElemTy = ElemQualTy.getTypePtr();
            ArrayTy = ElemTy->getAsArrayTypeUnsafe();
          } while (ArrayTy);
        }
        revng_assert(ElemTy);

        // If we have a field which is an array of structs, recur
        if (auto *ElemStructType = ElemTy->getAs<clang::RecordType>()) {
          rc_recur addDeclsInOrder(ElemStructType->getDecl(),
                                   OrderedDecls,
                                   NeedForwardDecl);
          continue;
        }

        // If the element is not a RecordType, it may stil be a pointer.
        PtrTy = ElemTy->getAs<clang::PointerType>();
      }

      // The field was not a struct, not an array, nor a pointer.
      // We have nothing to do.
      if (not PtrTy)
        continue;

      clang::QualType PointeeQualType = PtrTy->getPointeeType();
      const clang::Type *PointeeType = PointeeQualType.getTypePtr();
      auto *PointeeStructType = PointeeType->getAs<clang::RecordType>();
      if (not PointeeStructType)
        continue;

      clang::RecordDecl *PointeeStructDecl = PointeeStructType->getDecl();
      revng_log(TypeDeclOrderLog,
                "field has type: " << PointeeStructDecl->getNameAsString()
                                   << '*');

      if (OrderedDecls.count(PointeeStructDecl)) {
        revng_log(TypeDeclOrderLog, "already declared");
        continue;
      }

      bool New = NeedForwardDecl.insert(PointeeStructDecl);
      if (New)
        revng_log(TypeDeclOrderLog, "FORWARD DECLARED");
      else
        revng_log(TypeDeclOrderLog, "already forward declared");
    }
  }

  revng_log(TypeDeclOrderLog, "INSERTED");
  OrderedDecls.insert(StructDecl);
};

void Decompiler::HandleTranslationUnit(ASTContext &Context) {
  revng_assert(not TheF.isDeclaration());

  auto FTags = FunctionTags::TagsSet::from(&TheF);
  revng_assert(FTags.contains(FunctionTags::Isolated));

  DeclCreator Declarator(ValueLayouts);

  IR2AST::StmtBuilder ASTBuilder(Context,
                                 Mark,
                                 ValueLayouts,
                                 SE,
                                 BlockToPHIIncoming,
                                 Declarator);

  Declarator.createTypeDeclsForFunctionPrototype(Context, &TheF);
  Declarator.createGlobalVarDeclUsedByFunction(Context, &TheF, ASTBuilder);
  Declarator.createFunctionAndCalleesDecl(Context, &TheF);

  clang::FunctionDecl *FunctionDecl = &Declarator.getFunctionDecl(&TheF);
  ASTBuilder.createAST(TheF, *FunctionDecl);

  clang::TranslationUnitDecl *TUDecl = Context.getTranslationUnitDecl();

  // Enforce proper ordering between dependent type declarations, and inject
  // forward type declarations when needed.
  RecordDeclSetVector OrderedDecls;
  RecordDeclSetVector NeedForwardDecl;
  for (const auto &TypeDecl : Declarator.types())
    if (clang::TypeDecl *TD = DeclCreator::getTypeDecl(TypeDecl))
      if (not OrderedDecls.count(cast<clang::RecordDecl>(TD)))
        addDeclsInOrder(TD, OrderedDecls, NeedForwardDecl);

  // Create and add forward-declare all the structs.
  for (const clang::RecordDecl *StructDecl : NeedForwardDecl)
    TUDecl->addDecl(clang::RecordDecl::Create(Context,
                                              StructDecl->getTagKind(),
                                              TUDecl,
                                              clang::SourceLocation{},
                                              clang::SourceLocation{},
                                              StructDecl->getIdentifier(),
                                              nullptr));

  // Create and add the definition of all the struct types, in an order that
  // guarantees that they are always well-defined.
  for (clang::RecordDecl *StructDecl : OrderedDecls)
    TUDecl->addDecl(StructDecl);

  for (const auto &[_, GDecl] : Declarator.globalDecls()) {
    if (FunctionDecl == GDecl)
      continue;
    TUDecl->addDecl(GDecl);
  }

  buildFunctionBody(&TheF, FunctionDecl, CombedAST, ASTBuilder, Mark);

  TUDecl->addDecl(FunctionDecl);

  using ConsumerPtr = std::unique_ptr<ASTConsumer>;
  ConsumerPtr Printer = CreateASTPrinter(std::move(Out), "");
  Printer->HandleTranslationUnit(Context);
}

std::unique_ptr<ASTConsumer> CDecompilerAction::newASTConsumer() {
  return std::make_unique<Decompiler>(F,
                                      CombedAST,
                                      BlockToPHIIncoming,
                                      LayoutMap,
                                      SE,
                                      Mark,
                                      std::move(O));
}

std::unique_ptr<ASTConsumer>
CDecompilerAction::CreateASTConsumer(CompilerInstance &, llvm::StringRef) {
  return newASTConsumer();
}

} // end namespace tooling
} // end namespace clang
