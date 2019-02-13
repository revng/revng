/// \brief DataFlow analysis to build the AST for a Function

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// LLVM includes
#include <llvm/IR/Instruction.h>

// clang includes
#include <clang/AST/ASTContext.h>
#include <clang/AST/Decl.h>
#include <clang/AST/DeclGroup.h>
#include <clang/AST/Expr.h>
#include <clang/AST/OperationKinds.h>
#include <clang/AST/Stmt.h>

// revng includes
#include <revng/Support/IRHelpers.h>

#include "ASTBuildAnalysis.h"

#include "IRASTTypeTranslation.h"
#include "Mangling.h"

static Logger<> ASTBuildLog("ast-builder");

using namespace llvm;
using namespace clang;
using ClangType = clang::Type;
using ClangPointerType = clang::PointerType;
using LLVMType = llvm::Type;
using LLVMPointerType = llvm::PointerType;

namespace IR2AST {

void Analysis::initInternal() {
  NVar = 0;
  ASTInfo = SerializationInfo();
}

using IncomingIDSet = SmallSet<unsigned, 8>;

using OneToSetIncomingPair = std::pair<unsigned, IncomingIDSet>;
using OneToSetIncomingMap = SmallVector<OneToSetIncomingPair, 8>;

static bool smallerSizeOneToSetIncomingPair(const OneToSetIncomingPair &P,
                                            const OneToSetIncomingPair &Q) {
  return P.second.size() < Q.second.size();
}

struct Incoming {
  BasicBlock *Block;
  Value *Val;
};

using IncomingVec = SmallVector<Incoming, 8>;
using BlockPtrVec = SmallVector<BasicBlock *, 8>;
using IncomingCandidatesVec = SmallVector<BlockPtrVec, 8>;
using BlockToIncomingMap = SmallMap<BasicBlock *, IncomingIDSet, 8>;

struct IncomingCandidatesInfoTy {
  IncomingCandidatesVec IncomingCandidates;
  BlockToIncomingMap BlocksToIncoming;
};

using DomTree = DominatorTreeBase<BasicBlock, /* IsPostDom = */ false>;

static IncomingCandidatesInfoTy
getCandidatesInfo(const PHINode *ThePHI, const DomTree &DT) {

  unsigned NPred = ThePHI->getNumIncomingValues();
  revng_assert(NPred > 1);

  IncomingCandidatesInfoTy Res = {
    IncomingCandidatesVec(NPred, {}), // All the candidates are empty
    {} // The mapping of candidates to incomings is empty
  };

  for (unsigned K = 0; K < NPred; ++K) {
    Value *V = ThePHI->getIncomingValue(K);
    if (not isa<Instruction>(V) and not isa<Argument>(V)
        and not isa<Constant>(V))
      continue;

    BasicBlock *CandidateB = ThePHI->getIncomingBlock(K);

    BasicBlock *DefBlock = nullptr;
    if (auto *Inst = dyn_cast<Instruction>(V)) {
      DefBlock = Inst->getParent();
    } else {
      revng_assert(isa<Argument>(V) or isa<Constant>(V));
      BasicBlock *ParentEntryBlock = &CandidateB->getParent()->getEntryBlock();
      if (auto *Arg = dyn_cast<Argument>(V)) {
        BasicBlock *FunEntryBlock = &Arg->getParent()->getEntryBlock();
        revng_assert(FunEntryBlock == ParentEntryBlock);
      }
      DefBlock = ParentEntryBlock;
    }
    revng_assert(CandidateB != nullptr);
    revng_assert(DefBlock != nullptr);

    BasicBlock *B = nullptr;
    auto &Candidates = Res.IncomingCandidates[K];
    auto *DomNode = DT.getNode(CandidateB);
    revng_assert(DomNode != nullptr);
    do {
      B = DomNode->getBlock();
      Candidates.push_back(B);
      Res.BlocksToIncoming[B].insert(K);
      DomNode = DT.getNode(B)->getIDom();
    } while (DomNode != nullptr and DT.dominates(DefBlock, B));
  }

  for (unsigned K = 0; K < NPred; ++K) {
    auto &KCandidates = Res.IncomingCandidates[K];
    BasicBlock *CurrCandidate = KCandidates[0];
    for (unsigned H = 0; H < NPred; ++H) {
      if (K == H or ThePHI->getIncomingValue(K) == ThePHI->getIncomingValue(H))
        continue;
      BlockPtrVec &OtherCandidates = Res.IncomingCandidates[H];
      auto CandidateMatch = std::find(OtherCandidates.begin(),
                                      OtherCandidates.end(),
                                      CurrCandidate);

      auto CandidateIt = CandidateMatch;
      auto CandidateEnd = OtherCandidates.end();
      for (; CandidateIt != CandidateEnd; ++CandidateIt)
        Res.BlocksToIncoming.at(*CandidateIt).erase(K);
      if (CandidateMatch != OtherCandidates.end())
        OtherCandidates.erase(CandidateMatch, OtherCandidates.end());
    }
  }

  return Res;
}

static void computePHIVarAssignments(const PHINode *ThePHI,
                                     const DomTree &DT,
                                     BlockToPHIIncomingMap &AssignmentBlocks) {

  IncomingCandidatesInfoTy CandidatesInfo = getCandidatesInfo(ThePHI, DT);
  IncomingCandidatesVec &IncomingCandidates = CandidatesInfo.IncomingCandidates;
  BlockToIncomingMap &BlocksToIncoming = CandidatesInfo.BlocksToIncoming;

  unsigned NPred = IncomingCandidates.size();

  // Compute maximum number of valid candidates across all the incomings.
  // Its value is also used later to disable further processing whenever an
  // incoming has discarded MaxNumCandidates candidates
  size_t MaxNumCandidates = 0;
  for (unsigned K = 0; K < NPred; ++K) {
    Value *V = ThePHI->getIncomingValue(K);
    if (not isa<Instruction>(V) and not isa<Argument>(V))
      continue;
    MaxNumCandidates = std::max(MaxNumCandidates, IncomingCandidates[K].size());
  }
  ++MaxNumCandidates;
  revng_assert(MaxNumCandidates != 0);

  unsigned NumAssigned = 0;
  SmallVector<size_t, 8> NumDiscarded(NPred, 0);

  // Independently of all the other results, we can already assign all the
  // incomings that are not Instructions nor Arguments
  for (unsigned K = 0; K < NPred; ++K) {
    Value *V = ThePHI->getIncomingValue(K);
    if (not isa<Instruction>(V) and not isa<Argument>(V)) {
      NumDiscarded[K] = MaxNumCandidates; // this incoming is complete
      AssignmentBlocks[ThePHI->getIncomingBlock(K)][ThePHI] = K;
      ++NumAssigned;
    } else {
      auto &KCandidates = IncomingCandidates[K];
      if (KCandidates.size() == 1) {
        NumDiscarded[K] = MaxNumCandidates; // this incoming is complete
        AssignmentBlocks[KCandidates.back()][ThePHI] = K;
        ++NumAssigned;
      }
    }
  }

  for (size_t NDisc = 0; NDisc < MaxNumCandidates; ++NDisc) {

    OneToSetIncomingMap Broken;

    for (unsigned K = 0; K < NPred; ++K) {
      if (NumDiscarded[K] != NDisc)
        continue;

      Broken.push_back({ K, {} });

      auto &KCandidates = IncomingCandidates[K];

      for (unsigned H = 0; H < NPred; ++H) {
        if (H == K or NumDiscarded[H] != NDisc
            or ThePHI->getIncomingValue(K) == ThePHI->getIncomingValue(H))
          continue;

        // Assigning K breaks H if any of the valid Candidates for K is also a
        // valid candidate for H
        bool KBreaksH = true;
        for (BasicBlock *Candidate : KCandidates)
          if (BlocksToIncoming.at(Candidate).count(H))
            KBreaksH = true;

        if (KBreaksH) {
          Broken.back().second.insert(H);
        }
      }
    }

    std::sort(Broken.begin(), Broken.end(), smallerSizeOneToSetIncomingPair);

    for (const auto &P : Broken) {
      unsigned IncomingIdx = P.first;
      size_t &NDiscardedP = NumDiscarded[IncomingIdx];
      if (NDiscardedP != NDisc)
        continue;
      BlockPtrVec &PCandidates = IncomingCandidates[IncomingIdx];
      NDiscardedP = MaxNumCandidates; // this incoming is complete
      auto &BlockAssignments = AssignmentBlocks[PCandidates.back()];
      bool New = BlockAssignments.insert({ ThePHI, IncomingIdx }).second;
      revng_assert(not New);
      ++NumAssigned;
      // Remove all the candidates in PCandidates from all the other lists of
      // candidates for all the other incomings related to a different Value
      for (auto &Other : P.second) {
        BlockPtrVec &OtherCandidates = IncomingCandidates[Other];
        size_t OtherCandidatesPrevSize = OtherCandidates.size();
        for (BasicBlock *PCand : PCandidates) {
          auto It = std::find(OtherCandidates.begin(),
                              OtherCandidates.end(),
                              PCand);
          if (It != OtherCandidates.end()) {
            OtherCandidates.erase(It);
            break;
          }
        }
        size_t NewDiscarded = OtherCandidatesPrevSize - OtherCandidates.size();
        if (NewDiscarded != 0) {
          NumDiscarded[Other] += NewDiscarded;
          revng_assert(NumDiscarded[Other] < MaxNumCandidates);
        }
      }
    }
  }
  revng_assert(NumAssigned == NPred);
}

static bool needsLabel(const BasicBlock &) {
  return false;
}

void Analysis::computePHIVars() {

  DomTree DT;
  DT.recalculate(F);

  uint64_t BBId = 0;
  for (BasicBlock &BB : F) {
    if (needsLabel(BB)) {
      auto BBName = BB.getName();
      IdentifierInfo &Id = BBName.empty() ?
        ASTCtx.Idents.get("unnamed_bb_" + std::to_string(BBId++)) :
        ASTCtx.Idents.get(makeCIdentifier(BBName));
      LabelDecl *Label = LabelDecl::Create(ASTCtx, &FDecl, {}, &Id);
      ASTInfo.LabelDecls[&BB] = Label;
    }
    for (Instruction &I : BB) {
      if (PHINode *ThePHI = dyn_cast<PHINode>(&I)) {
        createVarDecl(ThePHI);
        computePHIVarAssignments(ThePHI, DT, ASTInfo.BlockToPHIIncoming);
      }
    }
  }
}

void Analysis::initialize() {
  Base::initialize();
  initInternal();
  computePHIVars();
}

static Expr *getParenthesizedExprForValue(Value *V,
                                          GlobalsMap &GlobalVarAST,
                                          FunctionsMap &FunctionAST,
                                          clang::ASTContext &ASTCtx,
                                          SerializationInfo &ASTInfo) {
  Expr *Res = getExprForValue(V, GlobalVarAST, FunctionAST, ASTCtx, ASTInfo);
  if (isa<clang::BinaryOperator>(Res) or isa<ConditionalOperator>(Res))
    Res = new (ASTCtx) ParenExpr({}, {}, Res);
  return Res;
}

static CStyleCastExpr *
createCast(QualType LHSQualTy, Expr *RHS, ASTContext &ASTCtx) {
  QualType RHSQualTy = RHS->getType();
  const ClangType *LHSTy = LHSQualTy.getTypePtr();
  const ClangType *RHSTy = RHSQualTy.getTypePtr();

  CastKind CK;
  if (LHSTy->isIntegerType()) {
    if (RHSTy->isIntegerType()) {
      CK = CastKind::CK_IntegralCast;
    } else if (RHSTy->isPointerType()) {
      CK = CastKind::CK_PointerToIntegral;
    } else {
      revng_abort();
    }
  } else if (LHSTy->isPointerType()) {
    if (RHSTy->isIntegerType()) {

      uint64_t PtrSize = ASTCtx.getTypeSize(LHSQualTy);
      uint64_t IntegerSize = ASTCtx.getTypeSize(RHSQualTy);
      revng_assert(PtrSize >= IntegerSize);
      if (PtrSize > IntegerSize)
        RHS = createCast(ASTCtx.getUIntPtrType(), RHS, ASTCtx);

      CK = CastKind::CK_IntegralToPointer;
    } else if (RHSTy->isPointerType()) {
      CK = CastKind::CK_BitCast;
    } else {
      revng_abort();
    }
  } else {
    revng_abort();
  }
  TypeSourceInfo *TI = ASTCtx.CreateTypeSourceInfo(LHSQualTy);
  return CStyleCastExpr::Create(ASTCtx,
                                LHSQualTy,
                                VK_RValue,
                                CK,
                                RHS,
                                nullptr,
                                TI,
                                {},
                                {});
}

Stmt *Analysis::buildAST(Instruction &I) {
  revng_log(ASTBuildLog, "Build AST for" << dumpToString(&I));
  switch (I.getOpcode()) {
  // ---- SUPPORTED INSTRUCTIONS ----
  // Terminators
  case Instruction::Br: {
    auto *Branch = cast<BranchInst>(&I);
    if (Branch->isUnconditional()) {
      LabelDecl *Label = ASTInfo.LabelDecls.at(Branch->getSuccessor(0));
      GotoStmt *GoTo = new (ASTCtx) GotoStmt(Label, {}, {});
      return GoTo;
    } else {
      LabelDecl *Then = ASTInfo.LabelDecls.at(Branch->getSuccessor(0));
      LabelDecl *Else = ASTInfo.LabelDecls.at(Branch->getSuccessor(1));
      GotoStmt *GoToThen = new (ASTCtx) GotoStmt(Then, {}, {});
      GotoStmt *GoToElse = new (ASTCtx) GotoStmt(Else, {}, {});
      Expr *Cond = getExprForValue(Branch->getCondition());
      revng_log(ASTBuildLog, "GOT!");
      if (ASTBuildLog.isEnabled() and Cond)
        Cond->dump();
      if (Cond->isLValue())
        Cond = ImplicitCastExpr::Create(ASTCtx,
                                        Cond->getType(),
                                        CastKind::CK_LValueToRValue,
                                        Cond,
                                        nullptr,
                                        VK_RValue);
      return new (ASTCtx) IfStmt(ASTCtx,
                                 {},
                                 false,
                                 nullptr,
                                 nullptr,
                                 Cond,
                                 GoToThen,
                                 {},
                                 GoToElse);
    }
  }
  case Instruction::Ret: {
    // FIXME: handle returned values properly
    ReturnInst *Ret = cast<ReturnInst>(&I);
    Value *RetVal = Ret->getReturnValue();
    Expr *ReturnedExpr = RetVal ? getExprForValue(RetVal) : nullptr;
    return new (ASTCtx) ReturnStmt({}, ReturnedExpr, nullptr);
  }
  case Instruction::Switch: {
    revng_abort("switch instructions are not supported yet");
    auto *Switch = cast<SwitchInst>(&I);

    Value *Cond = Switch->getCondition();
    Expr *CondE = getExprForValue(Cond);

    SwitchStmt *S = new (ASTCtx) SwitchStmt(ASTCtx, nullptr, nullptr, CondE);

    unsigned NumCases = Switch->getNumCases() + 1; // +1 is for the default
    CompoundStmt *Body = CompoundStmt::CreateEmpty(ASTCtx, NumCases);

    BasicBlock *DefaultBlock = Switch->getDefaultDest();
    LabelDecl *DefaultLabel = ASTInfo.LabelDecls.at(DefaultBlock);
    GotoStmt *GoToDefault = new (ASTCtx) GotoStmt(DefaultLabel, {}, {});
    DefaultStmt *Default = new (ASTCtx) DefaultStmt({}, {}, GoToDefault);
    S->addSwitchCase(Default);

    int K = 0;

    for (auto CIt : Switch->cases()) {
      BasicBlock *CaseBlock = CIt.getCaseSuccessor();
      if (CaseBlock == DefaultBlock)
        continue;
      ConstantInt *CaseVal = CIt.getCaseValue();
      Expr *CaseCond = getExprForValue(CaseVal);
      LabelDecl *CaseLabel = ASTInfo.LabelDecls.at(CaseBlock);
      GotoStmt *GoToCase = new (ASTCtx) GotoStmt(CaseLabel, {}, {});
      CaseStmt *Case = new (ASTCtx) CaseStmt(CaseCond, nullptr, {}, {}, {});
      Case->setSubStmt(GoToCase);
      S->addSwitchCase(Case);
      Body->body_begin()[K++] = Case;
    }

    Body->body_begin()[K] = Default;
    S->setBody(Body);
    return S;
  }
  // Standard binary operators
  case Instruction::Add:
  case Instruction::Sub:
  case Instruction::Mul:
  // Standard division operators (with signedness)
  case Instruction::UDiv:
  case Instruction::SDiv:
  case Instruction::URem:
  case Instruction::SRem:
  // Logical operators
  case Instruction::And:
  case Instruction::Or:
  case Instruction::Xor:
  // Other instructions
  case Instruction::ICmp:
  case Instruction::Shl:
  case Instruction::LShr:
  case Instruction::AShr: {
    return createRValueExprForBinaryOperator(I);
  }
  // Memory instructions
  case Instruction::Alloca: {
    // TODO: for now we ignore the alignment of the alloca. This might turn out
    // not to be safe later, because it does not take into account the alignment
    // of future accesses in the `Alloca`ted space. If the code is then
    // recompiled for an architecture that does not support unaligned access
    // this may cause crashes.
    auto *Alloca = cast<AllocaInst>(&I);
    revng_assert(Alloca->isStaticAlloca());
    // First, create a VarDecl, for an array of char to place in the BasicBlock
    // where the AllocaInst is
    const DataLayout &DL = F.getParent()->getDataLayout();
    uint64_t AllocaSize = *Alloca->getAllocationSizeInBits(DL);
    revng_assert(AllocaSize <= std::numeric_limits<unsigned>::max());
    APInt ArraySize = APInt(32, static_cast<unsigned>(AllocaSize));
    using ArraySizeMod = clang::ArrayType::ArraySizeModifier;
    ArraySizeMod SizeMod = ArraySizeMod::Normal;
    QualType CharTy = ASTCtx.CharTy;
    QualType ArrayTy = ASTCtx.getConstantArrayType(CharTy,
                                                   ArraySize,
                                                   SizeMod,
                                                   0);
    const std::string VarName = "var_" + std::to_string(NVar++);
    IdentifierInfo &Id = ASTCtx.Idents.get(VarName);
    VarDecl *ArrayDecl = VarDecl::Create(ASTCtx,
                                         &FDecl,
                                         {},
                                         {},
                                         &Id,
                                         ArrayTy,
                                         nullptr,
                                         StorageClass::SC_None);
    FDecl.addDecl(ArrayDecl);
    ASTInfo.AllocaDecls[Alloca] = ArrayDecl;
    // Second, create an Expr for the address of the first element of the array.
    QualType CharPtrTy = ASTCtx.getPointerType(ASTCtx.CharTy);
    Expr *ArrayDeclRef = new (ASTCtx)
      DeclRefExpr(ArrayDecl, false, ArrayTy, VK_LValue, {});
    CastKind Kind = CastKind::CK_ArrayToPointerDecay;
    Expr *ArrayPtrDecay = ImplicitCastExpr::Create(ASTCtx,
                                                   CharPtrTy,
                                                   Kind,
                                                   ArrayDeclRef,
                                                   nullptr,
                                                   VK_RValue);
    Expr *ArrayIdx = IntegerLiteral::Create(ASTCtx,
                                            APInt::getNullValue(32),
                                            ASTCtx.IntTy,
                                            {});
    Expr *ArraySubscript = new (ASTCtx) ArraySubscriptExpr(ArrayPtrDecay,
                                                           ArrayIdx,
                                                           ASTCtx.CharTy,
                                                           VK_LValue,
                                                           OK_Ordinary,
                                                           {});
    using Unary = clang::UnaryOperator;
    return new (ASTCtx) Unary(ArraySubscript,
                              UnaryOperatorKind::UO_AddrOf,
                              CharPtrTy,
                              VK_RValue,
                              OK_Ordinary,
                              {},
                              false);
  }
  case Instruction::Load: {
    auto *Load = cast<LoadInst>(&I);
    Value *Addr = Load->getPointerOperand();
    Expr *AddrExpr = getParenthesizedExprForValue(Addr, GlobalVarAST, FunctionAST, ASTCtx, ASTInfo);
    revng_log(ASTBuildLog, "GOT!");
    if (ASTBuildLog.isEnabled() and AddrExpr)
      AddrExpr->dump();
    if (not isa<GlobalVariable>(Addr)) {
      QualType PointeeType = IRASTTypeTranslation::getQualType(Load, ASTCtx);

      QualType QualAddrType = AddrExpr->getType();
      const ClangType *AddrTy = QualAddrType.getTypePtr();
      if (not AddrTy->isPointerType()) {
        revng_assert(AddrTy->isBuiltinType());
        const BuiltinType *AddrExprTy = cast<BuiltinType>(AddrTy);
        revng_assert(AddrExprTy->isInteger());

        QualType PtrTy = ASTCtx.getPointerType(PointeeType);
        uint64_t PtrSize = ASTCtx.getTypeSize(PtrTy);
        uint64_t IntegerSize = ASTCtx.getTypeSize(AddrExprTy);
        revng_assert(PtrSize >= IntegerSize);
        if (PtrSize > IntegerSize)
          AddrExpr = createCast(ASTCtx.getUIntPtrType(), AddrExpr, ASTCtx);
        AddrExpr = createCast(PtrTy, AddrExpr, ASTCtx);
      }


      if (isa<llvm::ConstantPointerNull>(Addr)) {
        QualType QualPtrTy = AddrExpr->getType();
        const auto *PtrType = cast<ClangPointerType>(QualPtrTy.getTypePtr());
        QualType QualPointeeTy = PtrType->getPointeeType();
        QualPointeeTy.addVolatile();
        QualType PtrToVolatileTy = ASTCtx.getPointerType(QualPointeeTy);
        AddrExpr = createCast(PtrToVolatileTy, AddrExpr, ASTCtx);
      }

      using Unary = clang::UnaryOperator;
      return new (ASTCtx) Unary(AddrExpr,
                                UnaryOperatorKind::UO_Deref,
                                PointeeType,
                                VK_LValue,
                                OK_Ordinary,
                                {},
                                false);
    }
    return AddrExpr;
  }
  case Instruction::Store: {
    auto *Store = cast<StoreInst>(&I);
    Value *Stored = Store->getValueOperand();
    Expr *LHS = getParenthesizedExprForValue(Store, GlobalVarAST, FunctionAST, ASTCtx, ASTInfo);
    QualType LHSQualTy = LHS->getType();
    revng_log(ASTBuildLog, "GOT!");
    if (ASTBuildLog.isEnabled() and LHS)
      LHS->dump();
    Expr *RHS = getParenthesizedExprForValue(Stored, GlobalVarAST, FunctionAST, ASTCtx, ASTInfo);
    revng_log(ASTBuildLog, "GOT!");
    if (ASTBuildLog.isEnabled() and RHS)
      RHS->dump();

    if (RHS->getType() != LHSQualTy) {
      if (isa<clang::BinaryOperator>(RHS))
        RHS = new (ASTCtx) ParenExpr({}, {}, RHS);
      RHS = createCast(LHSQualTy, RHS, ASTCtx);
    }

    BinaryOperatorKind BinOpKind = BinaryOperatorKind::BO_Assign;
    return new (ASTCtx) clang::BinaryOperator(LHS,
                                              RHS,
                                              BinOpKind,
                                              LHSQualTy,
                                              VK_RValue,
                                              OK_Ordinary,
                                              {},
                                              FPOptions());
  }
  // Convert instructions
  case Instruction::Trunc:
  case Instruction::ZExt:
  case Instruction::SExt:
  case Instruction::IntToPtr:
  case Instruction::PtrToInt:
  case Instruction::BitCast: {
    revng_assert(I.getNumOperands() == 1);
    Expr *Res = getParenthesizedExprForValue(I.getOperand(0), GlobalVarAST, FunctionAST, ASTCtx, ASTInfo);
    QualType LHSQualType = IRASTTypeTranslation::getQualType(&I, ASTCtx);
    if (LHSQualType != Res->getType())
      Res = createCast(LHSQualType, Res, ASTCtx);
    revng_log(ASTBuildLog, "GOT!");
    if (ASTBuildLog.isEnabled() and Res)
      Res->dump();
    return Res;
  }
  // Other instructions
  case Instruction::Select: {
    Expr *Cond = getParenthesizedExprForValue(I.getOperand(0), GlobalVarAST, FunctionAST, ASTCtx, ASTInfo);
    revng_log(ASTBuildLog, "GOT!");
    if (ASTBuildLog.isEnabled() and Cond)
      Cond->dump();
    Expr *TrueExpr = getParenthesizedExprForValue(I.getOperand(1), GlobalVarAST, FunctionAST, ASTCtx, ASTInfo);
    revng_log(ASTBuildLog, "GOT!");
    if (ASTBuildLog.isEnabled() and TrueExpr)
      TrueExpr->dump();
    Expr *FalseExpr = getParenthesizedExprForValue(I.getOperand(2), GlobalVarAST, FunctionAST, ASTCtx, ASTInfo);
    revng_log(ASTBuildLog, "GOT!");
    if (ASTBuildLog.isEnabled() and FalseExpr)
      FalseExpr->dump();
    QualType ASTType = IRASTTypeTranslation::getQualType(&I, ASTCtx);
    return new (ASTCtx) ConditionalOperator(Cond,
                                            {},
                                            TrueExpr,
                                            {},
                                            FalseExpr,
                                            ASTType,
                                            VK_RValue,
                                            OK_Ordinary);
  }
  case Instruction::Call: {
    auto *TheCall = cast<CallInst>(&I);
    Function *CalleeFun = getCallee(TheCall);

    Expr *CalleeExpr = getExprForValue(CalleeFun);
    revng_log(ASTBuildLog, "GOT!");
    if (ASTBuildLog.isEnabled() and CalleeExpr)
      CalleeExpr->dump();

    size_t NumArgs = CalleeFun->arg_size();
    FunctionDecl *FD = FunctionAST.at(CalleeFun);
    size_t NumParms = FD->param_size();
    unsigned NumOps = TheCall->getNumArgOperands();
    bool HasNoParms = NumParms == 0
                      or (NumParms == 1
                          and FD->getParamDecl(0)->getType() == ASTCtx.VoidTy);
    revng_assert(HasNoParms or NumArgs == NumParms);
    revng_assert(NumArgs == NumOps);

    auto Args = SmallVector<Expr *, 8>(NumOps, nullptr);
    if (not HasNoParms) {
      for (unsigned OpId = 0; OpId < NumOps; ++OpId) {
        Value *Operand = TheCall->getOperand(OpId);
        Expr *ArgExpr = getExprForValue(Operand);
        QualType ArgQualTy = ArgExpr->getType();

        ParmVarDecl *ParmDecl = FD->getParamDecl(OpId);
        QualType ParmQualTy = ParmDecl->getType();

        if (ParmQualTy != ArgQualTy) {
          ArgExpr = new (ASTCtx) ParenExpr({}, {}, ArgExpr);
          ArgExpr = createCast(ParmQualTy, ArgExpr, ASTCtx);
        }

        Args[OpId] = ArgExpr;
      }
    }

    QualType ReturnType = IRASTTypeTranslation::getQualType(TheCall, ASTCtx);
    return new (ASTCtx)
      CallExpr(ASTCtx, CalleeExpr, Args, ReturnType, VK_RValue, {});
  }
  case Instruction::Unreachable: {
    Function *AbortFun = F.getParent()->getFunction("abort");
    Expr *CalleeExpr = getExprForValue(AbortFun);
    SmallVector<Expr *, 8> Args;
    QualType ReturnType = ASTCtx.VoidTy;
    return new (ASTCtx)
      CallExpr(ASTCtx, CalleeExpr, Args, ReturnType, VK_RValue, {});
  }

  // ---- UNSUPPORTED INSTRUCTIONS ----
  // Terminators
  case Instruction::IndirectBr:
  case Instruction::Invoke:
  case Instruction::Resume:
  case Instruction::CleanupRet:
  case Instruction::CatchRet:
  case Instruction::CatchPad:
  case Instruction::CatchSwitch:
  // Memory instructions
  case Instruction::GetElementPtr:
  case Instruction::AtomicCmpXchg:
  case Instruction::AtomicRMW:
  case Instruction::Fence:
  // Binary operators for floats
  case Instruction::FAdd:
  case Instruction::FSub:
  case Instruction::FMul:
  case Instruction::FDiv:
  case Instruction::FRem:
  // Convert instructions
  case Instruction::FPTrunc:
  case Instruction::FPExt:
  case Instruction::FPToUI:
  case Instruction::FPToSI:
  case Instruction::UIToFP:
  case Instruction::SIToFP:
  case Instruction::AddrSpaceCast:
  // Other instructions
  case Instruction::PHI:
  case Instruction::FCmp:
  case Instruction::VAArg:
  case Instruction::ExtractElement:
  case Instruction::InsertElement:
  case Instruction::ShuffleVector:
  case Instruction::ExtractValue:
  case Instruction::InsertValue:
  case Instruction::LandingPad:
  case Instruction::CleanupPad:
  default:
    revng_abort("Unexpected operation");
  }
  revng_abort("Unexpected operation");
}

static bool isPure(const Instruction & /*Call*/) {
  return false;
}

Analysis::InterruptType Analysis::transfer(BasicBlock *BB) {
  LatticeElement PendingToSerialize = this->State[BB].copy();

  revng_log(ASTBuildLog,
            "BB in Function: " << BB->getParent()->getName() << '\n'
                               << BB);

  for (Instruction &I : *BB) {
    if (isa<PHINode>(&I))
      continue;
    // Skip this for now. We'll need to change it if we ever want to emit code
    // with goto statements
    if (isa<BranchInst>(&I))
      continue;
    Stmt *NewStmt = buildAST(I);
    ASTInfo.PendingExprs[&I] = NewStmt;
    revng_log(ASTBuildLog, "Add to Pending");
    if (ASTBuildLog.isEnabled() and NewStmt)
      NewStmt->dump();

    revng_log(ASTBuildLog, "Operands:");
    ASTBuildLog.indent();

    for (auto &TheUse : I.operands()) {
      ASTBuildLog.indent();
      Value *V = TheUse.get();
      if (auto *UseInstr = dyn_cast<Instruction>(V)) {
        revng_log(ASTBuildLog, "Op is Instruction: " << dumpToString(UseInstr));
        PendingToSerialize.erase(UseInstr);
      } else {
        revng_log(ASTBuildLog, "Op is NOT Instruction: " << dumpToString(V));
        revng_assert(isa<Argument>(V) or isa<Constant>(V) or isa<BasicBlock>(V)
                     or isa<MetadataAsValue>(V));
      }
      ASTBuildLog.unindent();
    }
    ASTBuildLog.unindent();

    bool HasSideEffects = isa<StoreInst>(&I)
                          or (isa<CallInst>(&I) and not isPure(I));
    if (HasSideEffects) {
      revng_log(ASTBuildLog, "Serialize Pending");
      markSetToSerialize(PendingToSerialize);
      PendingToSerialize = LatticeElement::top(); // empty set
      markValueToSerialize(&I);
    } else {
      switch (I.getNumUses()) {
      case 1:
        if (isa<BranchInst>(I.uses().begin()->getUser()))
          markValueToSerialize(&I);
        else
          PendingToSerialize.insert(&I);
        break;
      default:
        revng_log(ASTBuildLog, "Mark this to serialize");
        markValueToSerialize(&I);
        break;
      }
    }
  }

  return InterruptType::createInterrupt(std::move(PendingToSerialize));
}

DeclMap::iterator Analysis::createVarDecl(Instruction *I) {
  revng_assert(ASTInfo.VarDecls.count(I) == 0);
  QualType ASTType = IRASTTypeTranslation::getQualType(I, ASTCtx);
  revng_assert(not ASTType.isNull());
  const std::string VarName = "var_" + std::to_string(NVar++);
  IdentifierInfo &Id = ASTCtx.Idents.get(VarName);
  VarDecl *NewVar = VarDecl::Create(ASTCtx,
                                    &FDecl,
                                    {},
                                    {},
                                    &Id,
                                    ASTType,
                                    nullptr,
                                    StorageClass::SC_None);
  FDecl.addDecl(NewVar);
  return ASTInfo.VarDecls.insert({ I, NewVar }).first;
}

void Analysis::markValueToSerialize(Instruction *I) {
  using DeclMap = std::map<llvm::Instruction *, clang::VarDecl *>;
  revng_assert(ASTInfo.InstrStmts.count(I) == 0);
  DeclMap &VarDecls = ASTInfo.VarDecls;
  DeclMap::iterator VarDeclIt = VarDecls.end();
  unsigned NUses = I->getNumUses();

  if (NUses) {
    // If the value has more than one use we need to create a variable for it,
    // but only if we haven't already
    VarDeclIt = VarDecls.find(I);
    if (VarDeclIt == VarDecls.end()) {
      VarDeclIt = createVarDecl(I);
      revng_log(ASTBuildLog, "Created VarDecl");
      if (ASTBuildLog.isEnabled())
        VarDeclIt->second->dump();
    }
  }

  revng_assert(VarDecls.find(I) == VarDecls.end() or NUses);
  Stmt *Result = nullptr;
  auto &Pending = ASTInfo.PendingExprs;
  auto InstrExprIt = Pending.find(I);
  revng_assert(InstrExprIt != Pending.end());
  Stmt *InstrExpr = InstrExprIt->second;
  if (VarDeclIt != VarDecls.end()) {
    BinaryOperatorKind BinOpKind = BinaryOperatorKind::BO_Assign;
    VarDecl *LHSVDecl = VarDeclIt->second;
    QualType LHSType = LHSVDecl->getType();
    Expr *LHS = new (ASTCtx)
      DeclRefExpr(LHSVDecl, false, LHSType, VK_LValue, {});

    Expr *RHS = cast<Expr>(InstrExpr);

    if (RHS->getType() != LHSType) {
      if (isa<clang::BinaryOperator>(RHS))
        RHS = new (ASTCtx) ParenExpr({}, {}, RHS);
      RHS = createCast(LHSType, RHS, ASTCtx);
    }

    Stmt *Assign = new (ASTCtx) clang::BinaryOperator(LHS,
                                                      RHS,
                                                      BinOpKind,
                                                      LHSType,
                                                      VK_RValue,
                                                      OK_Ordinary,
                                                      {},
                                                      FPOptions());
    Result = Assign;
  } else {
    Result = InstrExpr;
  }
  revng_log(ASTBuildLog, "Remove From Pending:");
  if (ASTBuildLog.isEnabled())
    InstrExpr->dump();
  Pending.erase(InstrExprIt);
  ASTInfo.InstrStmts[I] = Result;
}

static clang::BinaryOperatorKind getClangBinaryOpKind(const Instruction &I) {
  clang::BinaryOperatorKind Res;
  switch (I.getOpcode()) {
  case Instruction::Add: {
    Res = clang::BinaryOperatorKind::BO_Add;
  } break;
  case Instruction::Sub: {
    Res = clang::BinaryOperatorKind::BO_Sub;
  } break;
  case Instruction::Mul: {
    Res = clang::BinaryOperatorKind::BO_Mul;
  }
  case Instruction::And: {
    Res = clang::BinaryOperatorKind::BO_And;
  } break;
  case Instruction::Or: {
    Res = clang::BinaryOperatorKind::BO_Or;
  } break;
  case Instruction::Xor: {
    Res = clang::BinaryOperatorKind::BO_Xor;
  } break;
  case Instruction::ICmp: {
    auto *CompareI = cast<CmpInst>(&I);
    switch (CompareI->getPredicate()) {
    case CmpInst::ICMP_EQ: {
      Res = clang::BinaryOperatorKind::BO_EQ;
    } break;
    case CmpInst::ICMP_NE: {
      Res = clang::BinaryOperatorKind::BO_NE;
    } break;
    case CmpInst::ICMP_UGT:
    case CmpInst::ICMP_SGT: {
      Res = clang::BinaryOperatorKind::BO_GT;
    } break;
    case CmpInst::ICMP_UGE:
    case CmpInst::ICMP_SGE: {
      Res = clang::BinaryOperatorKind::BO_GE;
    } break;
    case CmpInst::ICMP_ULT:
    case CmpInst::ICMP_SLT: {
      Res = clang::BinaryOperatorKind::BO_LT;
    } break;
    case CmpInst::ICMP_ULE:
    case CmpInst::ICMP_SLE: {
      Res = clang::BinaryOperatorKind::BO_LE;
    } break;
    default:
      revng_abort("Unsupported comparison operator");
    }
  } break;
  case Instruction::Shl: {
    Res = clang::BinaryOperatorKind::BO_Shl;
  } break;
  case Instruction::LShr:
  case Instruction::AShr: {
    Res = clang::BinaryOperatorKind::BO_Shr;
  } break;
  case Instruction::UDiv:
  case Instruction::SDiv: {
    Res = clang::BinaryOperatorKind::BO_Div;
  } break;
  case Instruction::URem:
  case Instruction::SRem: {
    Res = clang::BinaryOperatorKind::BO_Rem;
  } break;
  default: {
    revng_log(ASTBuildLog, "Unsupported operation" << dumpToString(&I) << '\n');
    revng_abort("Unsupported binary operator");
  }
  }
  return Res;
}

static std::pair<Expr *, Expr *> getCastedBinaryOperands(ASTContext &ASTCtx,
                                                         const Instruction &I,
                                                         Expr *Op0Expr,
                                                         Expr *Op1Expr) {
  std::pair<Expr *, Expr *> Res = std::make_pair(Op0Expr, Op1Expr);

  QualType LHSQualTy = Op0Expr->getType();
  QualType RHSQualTy = Op1Expr->getType();
  const ClangType *LHSTy = LHSQualTy.getTypePtr();
  const ClangType *RHSTy = RHSQualTy.getTypePtr();
  revng_assert(LHSTy->isIntegerType() and RHSTy->isIntegerType());
  uint64_t LHSSize = ASTCtx.getTypeSize(LHSTy);
  uint64_t RHSSize = ASTCtx.getTypeSize(RHSTy);
  unsigned OpCode = I.getOpcode();
  revng_assert(LHSSize == RHSSize or OpCode == Instruction::Shl
               or OpCode == Instruction::LShr or OpCode == Instruction::AShr);
  QualType SignedTy = ASTCtx.getIntTypeForBitwidth(LHSSize, /* Signed */ true);

  switch (OpCode) {
  case Instruction::Add:
  case Instruction::Sub:
  case Instruction::Mul:
  case Instruction::And:
  case Instruction::Or:
  case Instruction::Xor:
  case Instruction::UDiv:
  case Instruction::URem:
  case Instruction::Shl:
  case Instruction::LShr: {
    // These instructions have unsigned semantics in llvm IR.
    // We emit unsigned integers by default, so these operations do not need
    // any cast to preserve the semantics in C.
  } break;

  case Instruction::SDiv:
  case Instruction::SRem:
  case Instruction::AShr:
  case Instruction::ICmp: {
    if (OpCode != Instruction::ICmp or cast<CmpInst>(&I)->isSigned()) {
      if (RHSTy->isUnsignedIntegerType())
        Res.second = createCast(SignedTy, Res.second, ASTCtx);
      if (LHSTy->isUnsignedIntegerType())
        Res.first = createCast(SignedTy, Res.first, ASTCtx);
    }
  } break;

  default: {
    revng_log(ASTBuildLog, "Unsupported operation" << dumpToString(&I) << '\n');
    revng_abort("Unsupported binary operator");
  }
  }
  return Res;
}

Expr *Analysis::createRValueExprForBinaryOperator(Instruction &I) {
  revng_assert(I.getNumOperands() == 2);
  BinaryOperatorKind BinOpKind = getClangBinaryOpKind(I);

  Value *LHSVal = I.getOperand(0);
  Expr *LHS = getParenthesizedExprForValue(LHSVal, GlobalVarAST, FunctionAST, ASTCtx, ASTInfo);
  revng_log(ASTBuildLog, "GOT!");
  if (ASTBuildLog.isEnabled() and LHS)
    LHS->dump();
  if (LHS->isLValue())
    LHS = ImplicitCastExpr::Create(ASTCtx,
                                   LHS->getType(),
                                   CastKind::CK_LValueToRValue,
                                   LHS,
                                   nullptr,
                                   VK_RValue);

  Value *RHSVal = I.getOperand(1);
  Expr *RHS = getParenthesizedExprForValue(RHSVal, GlobalVarAST, FunctionAST, ASTCtx, ASTInfo);
  revng_log(ASTBuildLog, "GOT!");
  if (ASTBuildLog.isEnabled() and RHS)
    RHS->dump();
  if (RHS->isLValue())
    RHS = ImplicitCastExpr::Create(ASTCtx,
                                   RHS->getType(),
                                   CastKind::CK_LValueToRValue,
                                   RHS,
                                   nullptr,
                                   VK_RValue);

  std::tie(LHS, RHS) = getCastedBinaryOperands(ASTCtx, I, LHS, RHS);

  Expr *Res = new (ASTCtx) clang::BinaryOperator(LHS,
                                                 RHS,
                                                 BinOpKind,
                                                 LHS->getType(),
                                                 VK_RValue,
                                                 OK_Ordinary,
                                                 {},
                                                 FPOptions());

  unsigned OpCode = I.getOpcode();
  switch (OpCode) {
  case Instruction::SDiv:
  case Instruction::SRem:
  case Instruction::AShr:
  case Instruction::ICmp: {
    QualType ResType = IRASTTypeTranslation::getQualType(&I, ASTCtx);
    Res = new (ASTCtx) ParenExpr({}, {}, Res);
    Res = createCast(ResType, Res, ASTCtx);
  } break;
  default:
    break;
  }
  return Res;
}

static Expr *getLiteralFromConstant(Constant *C,
                                    GlobalsMap &GlobalVarAST,
                                    FunctionsMap &FunctionAST,
                                    clang::ASTContext &ASTCtx,
                                    SerializationInfo &ASTInfo);

Expr *getExprForValue(Value *V,
                      GlobalsMap &GlobalVarAST,
                      FunctionsMap &FunctionAST,
                      clang::ASTContext &ASTCtx,
                      SerializationInfo &ASTInfo) {
  revng_log(ASTBuildLog, "getExprForValue: " << dumpToString(V));
  if (isa<ConstantData>(V) or isa<ConstantExpr>(V)) {
    return getLiteralFromConstant(cast<Constant>(V), GlobalVarAST, FunctionAST, ASTCtx, ASTInfo);
  } else if (auto *F = dyn_cast<Function>(V)) {
    FunctionDecl *FDecl = FunctionAST.at(F);
    QualType Type = FDecl->getType();
    DeclRefExpr *Res = new (ASTCtx)
      DeclRefExpr(FDecl, false, Type, VK_LValue, {});
    return Res;
  } else if (auto *G = dyn_cast<GlobalVariable>(V)) {
    VarDecl *GlobalVarDecl = GlobalVarAST.at(G);
    QualType Type = GlobalVarDecl->getType();
    DeclRefExpr *Res = new (ASTCtx)
      DeclRefExpr(GlobalVarDecl, false, Type, VK_LValue, {});
    return Res;
  } else if (auto *I = dyn_cast<Instruction>(V)) {

    auto &Pending = ASTInfo.PendingExprs;
    auto InstrExprIt = Pending.find(I);
    if (InstrExprIt != Pending.end()) {
      // If the Instruction has an entry in Pending, it means that it has not
      // been marked for serialization yet, hence the Expr for that Value should
      // simply be what we've found in Pending.

      return cast<Expr>(InstrExprIt->second);
    }

    // For all the other instructions that have already been marked for
    // serialization we should have an associated entry in VarDecl.
    // We simply return a DeclRefExpr wrapping the VarDecl associated with I.
    auto &VarDecls = ASTInfo.VarDecls;
    auto VarDeclIt = VarDecls.find(I);
    if (VarDeclIt != VarDecls.end()) {
      revng_assert(VarDeclIt->second != nullptr);
      VarDecl *VDecl = VarDeclIt->second;
      QualType Type = VDecl->getType();
      DeclRefExpr *Res = new (ASTCtx)
        DeclRefExpr(VDecl, false, Type, VK_LValue, {});
      return Res;
    }

    // If we don't have a VarDecl associated with I
    if (isa<LoadInst>(I) or isa<StoreInst>(I)) {
      // Load and Store Instruction are serialized as ExprLHS = ExprRHS.
      // getExprForValue returns the ExprLHS.
      auto *Store = dyn_cast<StoreInst>(I);
      auto *Load = dyn_cast<LoadInst>(I);

      Value *Addr = nullptr;
      if (Load)
        Addr = Load->getPointerOperand();
      else
        Addr = Store->getPointerOperand();

      Expr *AddrExpr = getParenthesizedExprForValue(Addr, GlobalVarAST, FunctionAST, ASTCtx, ASTInfo);
      revng_log(ASTBuildLog, "GOT!");
      if (ASTBuildLog.isEnabled() and AddrExpr)
        AddrExpr->dump();
      // If we're moving from or into a GlobalVariable ExprLHS is just aì
      // DeclRefExpr for that GlobalVariable
      if (isa<GlobalVariable>(Addr))
        return AddrExpr;

      // Otherwise ExprLHS dereferences AddrExpr
      QualType QualAddrType = AddrExpr->getType();
      AddrExpr = ImplicitCastExpr::Create(ASTCtx,
                                          QualAddrType,
                                          CastKind::CK_LValueToRValue,
                                          AddrExpr,
                                          nullptr,
                                          VK_RValue);

      QualType PointeeType;
      if (Load) {
        PointeeType = IRASTTypeTranslation::getQualType(Load, ASTCtx);
      } else {
        Value *Stored = Store->getValueOperand();
        PointeeType = IRASTTypeTranslation::getQualType(Stored, ASTCtx);
      }

      QualAddrType = AddrExpr->getType();
      const ClangType *AddrTy = QualAddrType.getTypePtr();
      if (not AddrTy->isPointerType()) {
        revng_assert(AddrTy->isBuiltinType());
        const BuiltinType *AddrExprTy = cast<BuiltinType>(AddrTy);
        revng_assert(AddrExprTy->isInteger());

        QualType PtrTy = ASTCtx.getPointerType(PointeeType);
        uint64_t PtrSize = ASTCtx.getTypeSize(PtrTy);
        uint64_t IntegerSize = ASTCtx.getTypeSize(AddrExprTy);
        revng_assert(PtrSize >= IntegerSize);
        if (PtrSize > IntegerSize)
          AddrExpr = createCast(ASTCtx.getUIntPtrType(), AddrExpr, ASTCtx);
        AddrExpr = createCast(PtrTy, AddrExpr, ASTCtx);
      }

      if (isa<llvm::ConstantPointerNull>(Addr)) {
        QualType QualPtrTy = AddrExpr->getType();
        const auto *PtrType = cast<ClangPointerType>(QualPtrTy.getTypePtr());
        QualType QualPointeeTy = PtrType->getPointeeType();
        QualPointeeTy.addVolatile();
        QualType PtrToVolatileTy = ASTCtx.getPointerType(QualPointeeTy);
        AddrExpr = createCast(PtrToVolatileTy, AddrExpr, ASTCtx);
      }

      using Unary = clang::UnaryOperator;
      return new (ASTCtx) Unary(AddrExpr,
                                UnaryOperatorKind::UO_Deref,
                                PointeeType,
                                VK_LValue,
                                OK_Ordinary,
                                {},
                                false);
    }
    if (auto *Cast = dyn_cast<CastInst>(I)) {
      Value *RHS = Cast->getOperand(0);
      Expr *Result = getParenthesizedExprForValue(RHS, GlobalVarAST, FunctionAST, ASTCtx, ASTInfo);
      LLVMType *RHSTy = Cast->getSrcTy();
      LLVMType *LHSTy = Cast->getDestTy();
      if (RHSTy != LHSTy) {
        revng_assert(RHSTy->isIntOrPtrTy() and LHSTy->isIntOrPtrTy());
        QualType DestTy = IRASTTypeTranslation::getQualType(LHSTy, ASTCtx);
        CastKind CK;
        switch (Cast->getOpcode()) {
        case Instruction::Trunc:
        case Instruction::ZExt:
        case Instruction::SExt: {
          revng_assert(not RHSTy->isPointerTy() and not LHSTy->isPointerTy());
          /// CK_IntegralCast - A cast between integral types (other than to
          /// boolean).  Variously a bitcast, a truncation, a sign-extension,
          /// or a zero-extension.
          ///    long l = 5;
          ///    (unsigned) i
          /// CAST_OPERATION(IntegralCast)
          CK = CastKind::CK_IntegralCast;
        } break;
        case Instruction::IntToPtr: {
          revng_assert(not RHSTy->isPointerTy() and LHSTy->isPointerTy());
          /// CK_IntegralToPointer - Integral to pointer.  A special kind of
          /// reinterpreting conversion.  Applies to normal, ObjC, and block
          /// pointers.
          ///    (char*) 0x1001aab0
          ///    reinterpret_cast<int*>(0)
          /// CAST_OPERATION(IntegralToPointer)
          QualType IntQualType = Result->getType();
          const ClangType *PtrType = DestTy.getTypePtr();
          revng_assert(PtrType->isPointerType());
          uint64_t PtrSize = ASTCtx.getTypeSize(DestTy);
          uint64_t IntegerSize = ASTCtx.getTypeSize(IntQualType);
          revng_assert(PtrSize >= IntegerSize);
          if (PtrSize > IntegerSize)
            Result = createCast(ASTCtx.getUIntPtrType(), Result, ASTCtx);
          CK = CastKind::CK_IntegralToPointer;
        } break;
        case Instruction::PtrToInt: {
          revng_assert(RHSTy->isPointerTy() and not LHSTy->isPointerTy());
          /// CK_PointerToIntegral - Pointer to integral.  A special kind of
          /// reinterpreting conversion.  Applies to normal, ObjC, and block
          /// pointers.
          ///    (intptr_t) "help!"
          /// CAST_OPERATION(PointerToIntegral)
          CK = CastKind::CK_PointerToIntegral;
        } break;
        case Instruction::BitCast: {
          revng_assert(RHSTy->isPointerTy() and LHSTy->isPointerTy());
          /// CK_BitCast - A conversion which causes a bit pattern of one type
          /// to be reinterpreted as a bit pattern of another type.  Generally
          /// the operands must have equivalent size and unrelated types.
          ///
          /// The pointer conversion char* -> int* is a bitcast.  A conversion
          /// from any pointer type to a C pointer type is a bitcast unless
          /// it's actually BaseToDerived or DerivedToBase.  A conversion to a
          /// block pointer or ObjC pointer type is a bitcast only if the
          /// operand has the same type kind; otherwise, it's one of the
          /// specialized casts below.
          ///
          /// Vector coercions are bitcasts.
          /// CAST_OPERATION(BitCast)
          CK = CastKind::CK_BitCast;
        } break;
        case Instruction::FPTrunc:
        case Instruction::FPExt:
        case Instruction::FPToUI:
        case Instruction::FPToSI:
        case Instruction::UIToFP:
        case Instruction::SIToFP:
        case Instruction::AddrSpaceCast:
        default:
          revng_abort();
        }

        TypeSourceInfo *TI = ASTCtx.CreateTypeSourceInfo(DestTy);
        Result = CStyleCastExpr::Create(ASTCtx,
                                        DestTy,
                                        VK_RValue,
                                        CK,
                                        Result,
                                        nullptr,
                                        TI,
                                        {},
                                        {});
      }
      revng_assert(Result);
      revng_log(ASTBuildLog, "GOT!");
      if (ASTBuildLog.isEnabled())
        Result->dump();
      return Result;
    }
    revng_abort();
  } else if (auto *Arg =  dyn_cast<Argument>(V)) {
    llvm::Function *F = Arg->getParent();
    llvm::FunctionType *FType = F->getFunctionType();
    revng_assert(not FType->isVarArg());
    unsigned NumLLVMParams = FType->getNumParams();
    unsigned ArgNo = Arg->getArgNo();
    clang::FunctionDecl *FDecl = FunctionAST.at(F);
    unsigned DeclNumParams = FDecl->getNumParams();
    revng_assert(NumLLVMParams == DeclNumParams);
    clang::ParmVarDecl *ParamVDecl = FDecl->getParamDecl(ArgNo);
    QualType Type = ParamVDecl->getType();
    DeclRefExpr *Res = new (ASTCtx)
      DeclRefExpr(ParamVDecl, false, Type, VK_LValue, {});
    return Res;

    //return ImplicitCastExpr::Create(ASTCtx,
    //                                Param->getType(),
    //                                CastKind::CK_LValueToRValue,
    //                                Param,
    //                                nullptr,
    //                                VK_RValue);
  } else {
    revng_abort();
  }
}

static Expr *getLiteralFromConstant(Constant *C,
                                    GlobalsMap &GlobalVarAST,
                                    FunctionsMap &FunctionAST,
                                    clang::ASTContext &ASTCtx,
                                    SerializationInfo &ASTInfo) {
  if (auto *CD = dyn_cast<ConstantData>(C)) {
    if (auto *CInt = dyn_cast<ConstantInt>(CD)) {
      QualType LiteralTy = IRASTTypeTranslation::getQualType(CInt, ASTCtx);
      const clang::Type *UnderlyingTy = LiteralTy.getTypePtrOrNull();
      revng_assert(UnderlyingTy != nullptr);
      const BuiltinType *BuiltinTy = cast<BuiltinType>(UnderlyingTy);
      uint64_t ConstValue = CInt->getValue().getZExtValue();
      APInt Const = APInt(ASTCtx.getIntWidth(LiteralTy), ConstValue);
      switch (BuiltinTy->getKind()) {
      case BuiltinType::Char_U:
      case BuiltinType::UChar:
      case BuiltinType::Char_S:
      case BuiltinType::SChar: {
        using CharKind = CharacterLiteral::CharacterKind;
        return new (ASTCtx)
          CharacterLiteral(ConstValue, CharKind::Ascii, ASTCtx.CharTy, {});
      }
      case BuiltinType::UShort: {
        Expr *Literal = IntegerLiteral::Create(ASTCtx, Const, ASTCtx.UnsignedIntTy, {});
        return createCast(ASTCtx.UnsignedShortTy, Literal, ASTCtx);
      }
      case BuiltinType::Short: {
        Expr *Literal = IntegerLiteral::Create(ASTCtx, Const, ASTCtx.IntTy, {});
        return createCast(ASTCtx.ShortTy, Literal, ASTCtx);
      }
      case BuiltinType::UInt:
      case BuiltinType::ULong:
      case BuiltinType::ULongLong:
      case BuiltinType::Int:
      case BuiltinType::Long:
      case BuiltinType::LongLong:
        return IntegerLiteral::Create(ASTCtx, Const, LiteralTy, {});
      case BuiltinType::UInt128: {
        uint64_t U128ConstVal = CInt->getValue().getZExtValue();
        APInt U128Const = APInt(128, U128ConstVal);
        return IntegerLiteral::Create(ASTCtx,
                                      U128Const,
                                      ASTCtx.UnsignedInt128Ty,
                                      {});
      }
      case BuiltinType::Int128: {
        uint64_t I128ConstVal = CInt->getValue().getZExtValue();
        APInt I128Const = APInt(128, I128ConstVal);
        return IntegerLiteral::Create(ASTCtx, I128Const, ASTCtx.Int128Ty, {});
      }
      default:
        revng_abort();
      }
    } else if (isa<ConstantPointerNull>(CD)) {
      QualType UIntPtr = ASTCtx.getUIntPtrType();
      uint64_t UIntPtrSize = ASTCtx.getTypeSize(UIntPtr);
      return IntegerLiteral::Create(ASTCtx,
                                    APInt::getNullValue(UIntPtrSize),
                                    UIntPtr,
                                    {});
    }
    revng_abort();
  }
  if (auto *CE = dyn_cast<ConstantExpr>(C)) {
    Expr *Result = nullptr;
    switch (CE->getOpcode()) {
    case Instruction::Trunc:
    case Instruction::ZExt:
    case Instruction::SExt:
    case Instruction::IntToPtr:
    case Instruction::PtrToInt:
    case Instruction::BitCast: {
      Result = getExprForValue(cast<ConstantInt>(CE->getOperand(0)), GlobalVarAST, FunctionAST, ASTCtx, ASTInfo);
      revng_log(ASTBuildLog, "GOT!");
      revng_assert(Result);
      if (ASTBuildLog.isEnabled())
        Result->dump();
    } break;
    default:
      revng_abort();
    }
    return Result;
  }
  revng_abort();
}

Expr *Analysis::getExprForValue(Value *V) {
  return IR2AST::getExprForValue(V, GlobalVarAST, FunctionAST, ASTCtx, ASTInfo);
}

} // namespace IR2AST
