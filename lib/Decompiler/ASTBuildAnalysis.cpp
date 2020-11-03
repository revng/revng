/// \brief DataFlow analysis to build the AST for a Function

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// LLVM includes
#include <llvm/IR/DerivedTypes.h>
#include <llvm/IR/Instruction.h>
#include <llvm/IR/Instructions.h>

// clang includes
#include <clang/AST/ASTContext.h>
#include <clang/AST/Decl.h>
#include <clang/AST/DeclGroup.h>
#include <clang/AST/Expr.h>
#include <clang/AST/OperationKinds.h>
#include <clang/AST/Stmt.h>

// revng includes
#include <revng/Support/IRHelpers.h>

// local includes
#include "ASTBuildAnalysis.h"
#include "DecompilationHelpers.h"
#include "IRASTTypeTranslation.h"
#include "Mangling.h"

static Logger<> ASTBuildLog("ast-builder");

using namespace llvm;
using namespace clang;
using ClangType = clang::Type;
using ClangPointerType = clang::PointerType;
using LLVMType = llvm::Type;
using LLVMPointerType = llvm::PointerType;
using namespace IRASTTypeTranslation;

namespace IR2AST {

Expr *StmtBuilder::getParenthesizedExprForValue(Value *V) {
  Expr *Res = getExprForValue(V);
  if (isa<clang::BinaryOperator>(Res) or isa<ConditionalOperator>(Res))
    Res = new (ASTCtx) ParenExpr({}, {}, Res);
  return Res;
}

Stmt *StmtBuilder::buildStmt(Instruction &I) {
  revng_log(ASTBuildLog, "Build AST for" << dumpToString(&I));
  switch (I.getOpcode()) {
  //
  // ---- SUPPORTED INSTRUCTIONS ----
  //

  //
  // ---- Terminators ----
  //
  case Instruction::Br: {
    revng_abort("branch instructions are not supported yet");
    auto *Branch = cast<BranchInst>(&I);
    if (Branch->isUnconditional()) {
      LabelDecl *Label = BBLabelDecls.at(Branch->getSuccessor(0));
      GotoStmt *GoTo = new (ASTCtx) GotoStmt(Label, {}, {});
      return GoTo;
    } else {
      LabelDecl *Then = BBLabelDecls.at(Branch->getSuccessor(0));
      LabelDecl *Else = BBLabelDecls.at(Branch->getSuccessor(1));
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
      return IfStmt::Create(ASTCtx,
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

    // HACK: handle return instructions containing a `ConstantStruct` by
    //       emitting a `return void`
    if (RetVal && isa<ConstantStruct>(RetVal)) {
      return ReturnStmt::Create(ASTCtx, {}, nullptr, nullptr);
    }

    Expr *ReturnedExpr = RetVal ? getExprForValue(RetVal) : nullptr;
    return ReturnStmt::Create(ASTCtx, {}, ReturnedExpr, nullptr);
  }
  case Instruction::Switch: {
    revng_abort("switch instructions are not supported yet");
    auto *Switch = cast<SwitchInst>(&I);

    Value *Cond = Switch->getCondition();
    Expr *CondE = getExprForValue(Cond);

    SwitchStmt *S = SwitchStmt::Create(ASTCtx, nullptr, nullptr, CondE);

    unsigned NumCases = Switch->getNumCases() + 1; // +1 is for the default
    CompoundStmt *Body = CompoundStmt::CreateEmpty(ASTCtx, NumCases);

    BasicBlock *DefaultBlock = Switch->getDefaultDest();
    LabelDecl *DefaultLabel = BBLabelDecls.at(DefaultBlock);
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
      LabelDecl *CaseLabel = BBLabelDecls.at(CaseBlock);
      GotoStmt *GoToCase = new (ASTCtx) GotoStmt(CaseLabel, {}, {});
      CaseStmt *Case = CaseStmt::Create(ASTCtx, CaseCond, nullptr, {}, {}, {});
      Case->setSubStmt(GoToCase);
      S->addSwitchCase(Case);
      Body->body_begin()[K++] = Case;
    }

    Body->body_begin()[K] = Default;
    S->setBody(Body);
    return S;
  }
  //
  // ---- Standard binary operators ----
  //
  case Instruction::Add:
  case Instruction::Sub:
  case Instruction::Mul:
  //
  // ---- Standard division operators (with signedness) ----
  //
  case Instruction::UDiv:
  case Instruction::SDiv:
  case Instruction::URem:
  case Instruction::SRem:
  //
  // ---- Logical operators ----
  //
  case Instruction::And:
  case Instruction::Or:
  case Instruction::Xor:
  //
  // ---- Other instructions ----
  //
  case Instruction::ICmp:
  case Instruction::Shl:
  case Instruction::LShr:
  case Instruction::AShr: {
    return createRValueExprForBinaryOperator(I);
  }
  //
  // ---- Memory instructions ----
  //
  case Instruction::Alloca: {
    VarDecl *ArrayDecl = AllocaDecls.at(cast<AllocaInst>(&I));
    QualType ArrayTy = ArrayDecl->getType();
    // Create an Expr for the address of the first element of the array.
    QualType CharPtrTy = ASTCtx.getPointerType(ASTCtx.CharTy);
    Expr *ArrayDeclRef = new (ASTCtx)
      DeclRefExpr(ASTCtx, ArrayDecl, false, ArrayTy, VK_LValue, {});
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
    Expr *AddrExpr = getParenthesizedExprForValue(Addr);
    revng_log(ASTBuildLog, "GOT!");
    if (ASTBuildLog.isEnabled() and AddrExpr)
      AddrExpr->dump();
    if (not isa<GlobalVariable>(Addr)) {
      clang::DeclContext &TUDecl = *ASTCtx.getTranslationUnitDecl();
      QualType PointeeType = getOrCreateQualType(Load,
                                                 ASTCtx,
                                                 TUDecl,
                                                 TypeDecls,
                                                 FieldDecls);

      QualType QualAddrType = AddrExpr->getType();
      const ClangType *AddrTy = QualAddrType.getTypePtr();
      if (not AddrTy->isPointerType()) {
        revng_assert(AddrTy->isBuiltinType());
        revng_assert(AddrTy->isIntegerType());

        QualType PtrTy = ASTCtx.getPointerType(PointeeType);
        uint64_t PtrSize = ASTCtx.getTypeSize(PtrTy);
        uint64_t IntegerSize = ASTCtx.getTypeSize(AddrTy);
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
    if (isa<UndefValue>(Stored))
      return nullptr;
    Expr *LHS = getParenthesizedExprForValue(Store);
    QualType LHSQualTy = LHS->getType();
    revng_log(ASTBuildLog, "GOT!");
    if (ASTBuildLog.isEnabled() and LHS)
      LHS->dump();
    Expr *RHS = getParenthesizedExprForValue(Stored);
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
  //
  // ---- Convert instructions ----
  //
  case Instruction::Trunc:
  case Instruction::ZExt:
  case Instruction::SExt:
  case Instruction::IntToPtr:
  case Instruction::PtrToInt:
  case Instruction::BitCast: {
    revng_assert(I.getNumOperands() == 1);
    Expr *Res = getParenthesizedExprForValue(I.getOperand(0));
    clang::DeclContext &TUDecl = *ASTCtx.getTranslationUnitDecl();
    QualType LHSQualType = getOrCreateQualType(&I,
                                               ASTCtx,
                                               TUDecl,
                                               TypeDecls,
                                               FieldDecls);
    if (LHSQualType != Res->getType())
      Res = createCast(LHSQualType, Res, ASTCtx);
    revng_log(ASTBuildLog, "GOT!");
    if (ASTBuildLog.isEnabled() and Res)
      Res->dump();
    return Res;
  }
  // ---- Other instructions ----
  case Instruction::Select: {
    Expr *Cond = getParenthesizedExprForValue(I.getOperand(0));
    revng_log(ASTBuildLog, "GOT!");
    if (ASTBuildLog.isEnabled() and Cond)
      Cond->dump();
    Expr *TrueExpr = getParenthesizedExprForValue(I.getOperand(1));
    revng_log(ASTBuildLog, "GOT!");
    if (ASTBuildLog.isEnabled() and TrueExpr)
      TrueExpr->dump();
    Expr *FalseExpr = getParenthesizedExprForValue(I.getOperand(2));
    revng_log(ASTBuildLog, "GOT!");
    if (ASTBuildLog.isEnabled() and FalseExpr)
      FalseExpr->dump();
    clang::DeclContext &TUDecl = *ASTCtx.getTranslationUnitDecl();
    QualType ASTType = getOrCreateQualType(&I,
                                           ASTCtx,
                                           TUDecl,
                                           TypeDecls,
                                           FieldDecls);
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
    FunctionDecl *FD = FunctionDecls.at(CalleeFun);
    size_t NumParms = FD->param_size();
    unsigned NumOps = TheCall->getNumArgOperands();
    bool HasNoParms = NumParms == 0
                      or (NumParms == 1
                          and FD->getParamDecl(0)->getType() == ASTCtx.VoidTy);
    revng_assert(HasNoParms or NumArgs == NumParms);
    const bool IsVariadic = FD->isVariadic();
    if (not FD->isVariadic())
      revng_assert(NumArgs == NumOps);

    auto Args = SmallVector<Expr *, 8>(NumOps, nullptr);
    revng_assert(not(not HasNoParms and IsVariadic));
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

    if (IsVariadic) {
      for (unsigned OpId = 0; OpId < NumOps; ++OpId) {
        Value *Operand = TheCall->getOperand(OpId);
        Expr *ArgExpr = getExprForValue(Operand);

        Args[OpId] = ArgExpr;
      }
    }

    clang::DeclContext &TUDecl = *ASTCtx.getTranslationUnitDecl();
    QualType ReturnType = getOrCreateQualType(TheCall->getType(),
                                              TheCall->getCalledFunction(),
                                              ASTCtx,
                                              TUDecl,
                                              TypeDecls,
                                              FieldDecls);
    return CallExpr::Create(ASTCtx,
                            CalleeExpr,
                            Args,
                            ReturnType,
                            VK_RValue,
                            {});
  }
  case Instruction::Unreachable: {
    Function *AbortFun = I.getModule()->getFunction("abort");
    Expr *CalleeExpr = getExprForValue(AbortFun);
    SmallVector<Expr *, 8> Args;
    QualType ReturnType = ASTCtx.VoidTy;
    return CallExpr::Create(ASTCtx,
                            CalleeExpr,
                            Args,
                            ReturnType,
                            VK_RValue,
                            {});
  }
  //
  // ---- Instructions for struct manipulation ----
  //
  case Instruction::InsertValue: {
    InsertValueInst *Insert = cast<InsertValueInst>(&I);
    revng_assert(Insert->getNumIndices() == 1);
    Value *AggregateOp = Insert->getAggregateOperand();
    revng_assert(isa<UndefValue>(AggregateOp)
                 or isa<InsertValueInst>(AggregateOp)
                 or isa<ConstantStruct>(AggregateOp));
    llvm::Type *AggregateTy = AggregateOp->getType();
    clang::TypeDecl *StructTypeDecl = TypeDecls.at(AggregateTy);
    Expr *StructExpr = getExprForValue(Insert);
    clang::DeclContext &TUDecl = *ASTCtx.getTranslationUnitDecl();
    QualType InsertedTy = getOrCreateQualType(Insert->getType(),
                                              Insert->getFunction(),
                                              ASTCtx,
                                              TUDecl,
                                              TypeDecls,
                                              FieldDecls);
    unsigned Idx = *Insert->idx_begin();
    FieldDecl *FieldDecl = FieldDecls.at(StructTypeDecl)[Idx];
    clang::DeclarationName FieldDeclName = FieldDecl->getIdentifier();
    clang::DeclarationNameInfo FieldDeclNameInfo(FieldDeclName, {});
    auto DAP = DeclAccessPair::make(FieldDecl, FieldDecl->getAccess());
    clang::Expr *LHS = MemberExpr::Create(ASTCtx,
                                          StructExpr,
                                          /*isarrow*/ false,
                                          {},
                                          {},
                                          {},
                                          FieldDecl,
                                          DAP,
                                          FieldDeclNameInfo,
                                          /*TemplateArgs*/ nullptr,
                                          InsertedTy,
                                          VK_LValue,
                                          OK_Ordinary,
                                          NOUR_None);
    clang::Expr *RHS = getExprForValue(Insert->getInsertedValueOperand());
    BinaryOperatorKind BinOpKind = BinaryOperatorKind::BO_Assign;
    AdditionalStmts[&I].push_back(new (ASTCtx)
                                    clang::BinaryOperator(LHS,
                                                          RHS,
                                                          BinOpKind,
                                                          LHS->getType(),
                                                          VK_RValue,
                                                          OK_Ordinary,
                                                          {},
                                                          FPOptions()));
    if (isa<UndefValue>(AggregateOp))
      return nullptr;
    if (isa<ConstantStruct>(AggregateOp))
      return nullptr;
    return getExprForValue(AggregateOp);
  }
  case Instruction::ExtractValue: {
    ExtractValueInst *Extract = cast<ExtractValueInst>(&I);
    revng_assert(Extract->getNumIndices() == 1);
    Value *AggregateOp = Extract->getAggregateOperand();
    if (isa<UndefValue>(AggregateOp))
      return nullptr;
    revng_assert(isa<CallInst>(AggregateOp));
    llvm::Type *AggregateTy = AggregateOp->getType();
    clang::TypeDecl *StructTypeDecl = TypeDecls.at(AggregateTy);
    Expr *StructExpr = getExprForValue(AggregateOp);
    clang::DeclContext &TUDecl = *ASTCtx.getTranslationUnitDecl();
    QualType ExtractedTy = getOrCreateQualType(Extract,
                                               ASTCtx,
                                               TUDecl,
                                               TypeDecls,
                                               FieldDecls);
    unsigned Idx = *Extract->idx_begin();
    FieldDecl *ExtractedFDecl = FieldDecls.at(StructTypeDecl)[Idx];
    clang::DeclarationName FieldDeclName = ExtractedFDecl->getIdentifier();
    clang::DeclarationNameInfo FieldDeclNameInfo(FieldDeclName, {});
    return MemberExpr::Create(ASTCtx,
                              StructExpr,
                              /*isarrow*/ false,
                              {},
                              {},
                              {},
                              ExtractedFDecl,
                              DeclAccessPair::make(ExtractedFDecl,
                                                   ExtractedFDecl->getAccess()),
                              FieldDeclNameInfo,
                              /*TemplateArgs*/ nullptr,
                              ExtractedTy,
                              VK_RValue,
                              OK_Ordinary,
                              NOUR_None);
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
  case Instruction::LandingPad:
  case Instruction::CleanupPad:
  default:
    revng_abort("Unexpected operation");
  }
  revng_abort("Unexpected operation");
}

clang::VarDecl *
StmtBuilder::getOrCreateLoopStateVarDecl(clang::FunctionDecl &FDecl) {
  if (not LoopStateVarDecl) {
    IdentifierInfo &Id = ASTCtx.Idents.get("loop_state_var");
    LoopStateVarDecl = VarDecl::Create(ASTCtx,
                                       &FDecl,
                                       {},
                                       {},
                                       &Id,
                                       ASTCtx.UnsignedIntTy,
                                       nullptr,
                                       StorageClass::SC_None);
    FDecl.addDecl(LoopStateVarDecl);
  }
  revng_assert(LoopStateVarDecl != nullptr);
  return LoopStateVarDecl;
}

clang::VarDecl *
StmtBuilder::getOrCreateSwitchStateVarDecl(clang::FunctionDecl &FDecl) {
  if (not SwitchStateVarDecl) {
    IdentifierInfo &Id = ASTCtx.Idents.get("switch_state_var");
    QualType BoolTy = getOrCreateBoolQualType(ASTCtx, TypeDecls);
    SwitchStateVarDecl = VarDecl::Create(ASTCtx,
                                         &FDecl,
                                         {},
                                         {},
                                         &Id,
                                         BoolTy,
                                         nullptr,
                                         StorageClass::SC_None);
    FDecl.addDecl(SwitchStateVarDecl);
  }
  revng_assert(SwitchStateVarDecl != nullptr);
  return SwitchStateVarDecl;
}

void StmtBuilder::createAST(llvm::Function &F, clang::FunctionDecl &FDecl) {
  revng_log(ASTBuildLog,
            "Building AST for Instructions in Function " << F.getName());

  uint64_t BBId = 0;
  ReversePostOrderTraversal<Function *> RPOT(&F);
  for (BasicBlock *BB : RPOT) {
    revng_log(ASTBuildLog, "BB: " << BB->getName());

    {
      // Create labels for Basic Blocks. This could potentially be disabled if
      // we choose not to have the option to emit goto statements ever.
      IdentifierInfo &Id = ASTCtx.Idents.get("bb_" + std::to_string(BBId++));
      LabelDecl *Label = LabelDecl::Create(ASTCtx, &FDecl, {}, &Id);
      BBLabelDecls[BB] = Label;
    }

    for (Instruction &I : *BB) {
      // We don't build clang's AST expressions for PHINodes nor for
      // BranchInsts and SwitchInsts.
      // PHINodes are not expanded into expressions because they expand in a
      // local variable, that is assigned multiple times for all the incoming
      // Values of the PHINode.
      // For BranchInsts, we don't create AST right now, because the emission of
      // control flow statements in C is driven by the ASTTree
      if (isa<BranchInst>(&I))
        continue;

      // For SwitchInsts, we don't create AST right now, because the emission of
      // control flow statements in C is driven by the ASTTree
      if (isa<SwitchInst>(&I))
        continue;

      // Each PHINode has an associated VarDecl
      if (isa<PHINode>(&I)) {
        revng_assert(VarDecls.count(&I) == 0);
        VarDecl *NewVarDecl = createVarDecl(&I, FDecl);
        VarDecls[&I] = NewVarDecl;
        continue;
      }

      if (isa<AllocaInst>(&I)) {
        // TODO: for now we ignore the alignment of the alloca. This might turn
        // out not to be safe later, because it does not take into account the
        // alignment of future accesses in the `Alloca`ted space. If the code is
        // then recompiled for an architecture that does not support unaligned
        // access this may cause crashes.
        AllocaInst *Alloca = cast<AllocaInst>(&I);
        revng_assert(Alloca->isStaticAlloca());
        // First, create a VarDecl, for an array of char to place in the
        // BasicBlock where the AllocaInst is
        const DataLayout &DL = F.getParent()->getDataLayout();
        auto *AllocatedTy = Alloca->getAllocatedType();
        uint64_t AllocaSize = DL.getTypeAllocSize(AllocatedTy);

        revng_assert(AllocaSize <= std::numeric_limits<unsigned>::max());
        APInt ArraySize = APInt(32, static_cast<unsigned>(AllocaSize));
        using ArraySizeMod = clang::ArrayType::ArraySizeModifier;
        ArraySizeMod SizeMod = ArraySizeMod::Normal;
        QualType CharTy = ASTCtx.CharTy;
        QualType ArrayTy = ASTCtx.getConstantArrayType(CharTy,
                                                       ArraySize,
                                                       nullptr,
                                                       SizeMod,
                                                       0);
        const std::string VarName = std::string("local_")
                                    + (I.hasName() ?
                                         I.getName().str() :
                                         (std::string("var_")
                                          + std::to_string(NVar++)));
        IdentifierInfo &Id = ASTCtx.Idents.get(makeCIdentifier(VarName));
        VarDecl *ArrayDecl = VarDecl::Create(ASTCtx,
                                             &FDecl,
                                             {},
                                             {},
                                             &Id,
                                             ArrayTy,
                                             nullptr,
                                             StorageClass::SC_None);
        FDecl.addDecl(ArrayDecl);
        AllocaDecls[Alloca] = ArrayDecl;
      }

      if (auto *Insert = dyn_cast<InsertValueInst>(&I)) {
        revng_assert(VarDecls.count(&I) == 0);
        VarDecl *NewVarDecl = createVarDecl(&I, FDecl);
        VarDecls[&I] = NewVarDecl;

        Value *AggregateOp = Insert->getAggregateOperand();
        if (auto *CS = dyn_cast<ConstantStruct>(AggregateOp)) {
          std::vector<Expr *> StructOpExpr;
          for (auto &OperandUse : CS->operands()) {
            Value *Operand = OperandUse.get();
            Constant *OperandConst = cast<Constant>(Operand);
            clang::Expr *OperandExpr = nullptr;
            if (isa<UndefValue>(OperandConst)) {
              QualType IntT = ASTCtx.IntTy;
              OperandExpr = new (ASTCtx) ImplicitValueInitExpr(IntT);
            } else {
              OperandExpr = getLiteralFromConstant(OperandConst);
            }
            revng_assert(OperandExpr != nullptr);
            StructOpExpr.push_back(OperandExpr);
          }

          clang::Expr *ILE = new (ASTCtx)
            InitListExpr(ASTCtx, {}, StructOpExpr, {});
          NewVarDecl->setInit(ILE);
        }
      }

      Stmt *NewStmt = buildStmt(I);
      if (NewStmt == nullptr)
        continue;
      InstrStmts[&I] = NewStmt;

      if (not isa<InsertValueInst>(&I) and I.getNumUses() > 0
          and ToSerialize.count(&I)) {
        revng_assert(VarDecls.count(&I) == 0);
        VarDecl *NewVarDecl = createVarDecl(&I, FDecl);
        VarDecls[&I] = NewVarDecl;
      }
    }
  }
}

VarDecl *
StmtBuilder::createVarDecl(Instruction *I, clang::FunctionDecl &FDecl) {
  clang::DeclContext &TUDecl = *ASTCtx.getTranslationUnitDecl();
  QualType ASTType;
  if (auto *Call = dyn_cast<llvm::CallInst>(I)) {
    ASTType = getOrCreateQualType(Call->getType(),
                                  Call->getCalledFunction(),
                                  ASTCtx,
                                  TUDecl,
                                  TypeDecls,
                                  FieldDecls);
  } else if (auto *Insert = dyn_cast<llvm::InsertValueInst>(I)) {
    ASTType = getOrCreateQualType(Insert->getType(),
                                  Insert->getFunction(),
                                  ASTCtx,
                                  TUDecl,
                                  TypeDecls,
                                  FieldDecls);
  } else {
    revng_assert(not isa<llvm::StructType>(I->getType()));
    ASTType = getOrCreateQualType(I, ASTCtx, TUDecl, TypeDecls, FieldDecls);
  }

  revng_assert(not ASTType.isNull());
  const std::string VarName = I->hasName() ?
                                I->getName().str() :
                                (std::string("var_") + std::to_string(NVar++));
  IdentifierInfo &Id = ASTCtx.Idents.get(makeCIdentifier(VarName));
  VarDecl *NewVarDecl = VarDecl::Create(ASTCtx,
                                        &FDecl,
                                        {},
                                        {},
                                        &Id,
                                        ASTType,
                                        nullptr,
                                        StorageClass::SC_None);
  FDecl.addDecl(NewVarDecl);
  return NewVarDecl;
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
  } break;
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
    case CmpInst::BAD_ICMP_PREDICATE:
    case CmpInst::BAD_FCMP_PREDICATE:
    case CmpInst::FCMP_TRUE:
    case CmpInst::FCMP_FALSE:
    case CmpInst::FCMP_OEQ:
    case CmpInst::FCMP_ONE:
    case CmpInst::FCMP_OGE:
    case CmpInst::FCMP_OGT:
    case CmpInst::FCMP_OLE:
    case CmpInst::FCMP_OLT:
    case CmpInst::FCMP_ORD:
    case CmpInst::FCMP_UNO:
    case CmpInst::FCMP_UEQ:
    case CmpInst::FCMP_UNE:
    case CmpInst::FCMP_UGT:
    case CmpInst::FCMP_UGE:
    case CmpInst::FCMP_ULT:
    case CmpInst::FCMP_ULE:
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

static bool is128Int(ASTContext &ASTCtx, clang::Expr *E) {
  const clang::Type *T = E->getType().getTypePtr();
  const clang::Type *Int128T = ASTCtx.Int128Ty.getTypePtr();
  const clang::Type *UInt128T = ASTCtx.UnsignedInt128Ty.getTypePtr();
  return T == Int128T or T == UInt128T;
}

static std::pair<Expr *, Expr *> getCastedBinaryOperands(ASTContext &ASTCtx,
                                                         const Instruction &I,
                                                         Expr *LHS,
                                                         Expr *RHS) {
  std::pair<Expr *, Expr *> Res = std::make_pair(LHS, RHS);

  QualType LHSQualTy = LHS->getType();
  QualType RHSQualTy = RHS->getType();
  const ClangType *LHSTy = LHSQualTy.getTypePtr();
  const ClangType *RHSTy = RHSQualTy.getTypePtr();
  revng_assert(LHSTy->isIntegerType() and RHSTy->isIntegerType());
  uint64_t LHSSize = ASTCtx.getTypeSize(LHSTy);
  uint64_t RHSSize = ASTCtx.getTypeSize(RHSTy);
  unsigned OpCode = I.getOpcode();
  revng_assert(LHSSize == RHSSize or OpCode == Instruction::Shl
               or OpCode == Instruction::LShr or OpCode == Instruction::AShr
               or is128Int(ASTCtx, RHS) or is128Int(ASTCtx, LHS));
  unsigned Size = static_cast<unsigned>(std::max(LHSSize, RHSSize));
  QualType SignedTy = ASTCtx.getIntTypeForBitwidth(Size, /* Signed */ true);

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

Expr *StmtBuilder::createRValueExprForBinaryOperator(Instruction &I) {
  revng_assert(I.getNumOperands() == 2);
  BinaryOperatorKind BinOpKind = getClangBinaryOpKind(I);

  Value *LHSVal = I.getOperand(0);
  Expr *LHS = getParenthesizedExprForValue(LHSVal);
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
  Expr *RHS = getParenthesizedExprForValue(RHSVal);
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
    clang::DeclContext &TUDecl = *ASTCtx.getTranslationUnitDecl();
    QualType ResType = getOrCreateQualType(&I,
                                           ASTCtx,
                                           TUDecl,
                                           TypeDecls,
                                           FieldDecls);
    Res = new (ASTCtx) ParenExpr({}, {}, Res);
    Res = createCast(ResType, Res, ASTCtx);
  } break;
  default:
    break;
  }
  return Res;
}

Expr *StmtBuilder::getBoolLiteral(bool V) {
  QualType IntT = ASTCtx.IntTy;
  QualType BoolTy = getOrCreateBoolQualType(ASTCtx, TypeDecls);
  APInt Const = APInt(ASTCtx.getIntWidth(IntT), V ? 1 : 0, true);
  Expr *IntLiteral = IntegerLiteral::Create(ASTCtx, Const, IntT, {});
  return createCast(BoolTy, IntLiteral, ASTCtx);
}

Expr *StmtBuilder::getUIntLiteral(uint64_t U) {
  QualType UIntT = ASTCtx.UnsignedIntTy;
  APInt Const = APInt(ASTCtx.getIntWidth(UIntT), U);
  return IntegerLiteral::Create(ASTCtx, Const, UIntT, {});
}

Expr *StmtBuilder::getExprForValue(Value *V) {
  revng_log(ASTBuildLog, "getExprForValue: " << dumpToString(V));
  if (auto *Fun = dyn_cast<Function>(V)) {
    FunctionDecl *FunDecl = FunctionDecls.at(Fun);
    QualType Type = FunDecl->getType();
    DeclRefExpr *Res = new (ASTCtx)
      DeclRefExpr(ASTCtx, FunDecl, false, Type, VK_LValue, {});
    return Res;
  } else if (auto *G = dyn_cast<GlobalVariable>(V)) {
    VarDecl *GlobalVarDecl = GlobalDecls.at(G);
    QualType Type = GlobalVarDecl->getType();
    DeclRefExpr *Res = new (ASTCtx)
      DeclRefExpr(ASTCtx, GlobalVarDecl, false, Type, VK_LValue, {});
    return Res;
  } else if (isa<llvm::ConstantData>(V) or isa<llvm::ConstantExpr>(V)) {
    return getLiteralFromConstant(cast<llvm::Constant>(V));

  } else if (auto *I = dyn_cast<Instruction>(V)) {

    // For all the other instructions that have already been marked for
    // serialization we should have an associated entry in VarDecl.
    // We simply return a DeclRefExpr wrapping the VarDecl associated with I.
    auto VarDeclIt = VarDecls.find(I);
    if (VarDeclIt != VarDecls.end()) {
      revng_assert(VarDeclIt->second != nullptr);
      VarDecl *VDecl = VarDeclIt->second;
      QualType Type = VDecl->getType();
      DeclRefExpr *Res = new (ASTCtx)
        DeclRefExpr(ASTCtx, VDecl, false, Type, VK_LValue, {});
      return Res;
    }

    auto InstrStmtIt = InstrStmts.find(I);
    if (InstrStmtIt != InstrStmts.end()) {
      // If the Instruction has an entry in InstrStmts, it means that we have
      // already computed an expression for it, so we can directly use that.
      return cast<Expr>(InstrStmtIt->second);
    }

    // If we reach this point we are creating an expression for a new
    // Instruction. This should only happen for Load, Store and casts.

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

      Expr *AddrExpr = getParenthesizedExprForValue(Addr);
      revng_log(ASTBuildLog, "GOT!");
      if (ASTBuildLog.isEnabled() and AddrExpr)
        AddrExpr->dump();
      // If we're moving from or into a GlobalVariable ExprLHS is just
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

      clang::DeclContext &TUDecl = *ASTCtx.getTranslationUnitDecl();
      QualType PointeeType;
      if (Load) {
        PointeeType = getOrCreateQualType(Load,
                                          ASTCtx,
                                          TUDecl,
                                          TypeDecls,
                                          FieldDecls);
      } else {
        Value *Stored = Store->getValueOperand();
        PointeeType = getOrCreateQualType(Stored,
                                          ASTCtx,
                                          TUDecl,
                                          TypeDecls,
                                          FieldDecls);
      }

      QualAddrType = AddrExpr->getType();
      const ClangType *AddrTy = QualAddrType.getTypePtr();
      if (not AddrTy->isPointerType()) {
        revng_assert(AddrTy->isBuiltinType());
        revng_assert(AddrTy->isIntegerType());

        QualType PtrTy = ASTCtx.getPointerType(PointeeType);
        uint64_t PtrSize = ASTCtx.getTypeSize(PtrTy);
        uint64_t IntegerSize = ASTCtx.getTypeSize(AddrTy);
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
      Expr *Result = getParenthesizedExprForValue(RHS);
      LLVMType *RHSTy = Cast->getSrcTy();
      LLVMType *LHSTy = Cast->getDestTy();
      if (RHSTy != LHSTy) {
        revng_assert(RHSTy->isIntOrPtrTy() and LHSTy->isIntOrPtrTy());
        clang::DeclContext &TUDecl = *ASTCtx.getTranslationUnitDecl();
        QualType DestTy = getOrCreateQualType(LHSTy,
                                              nullptr,
                                              ASTCtx,
                                              TUDecl,
                                              TypeDecls,
                                              FieldDecls);
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
        case Instruction::CastOpsEnd:
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
  } else if (auto *Arg = dyn_cast<Argument>(V)) {
    llvm::Function *ArgFun = Arg->getParent();
    llvm::FunctionType *FType = ArgFun->getFunctionType();
    revng_assert(not FType->isVarArg());
    unsigned NumLLVMParams = FType->getNumParams();
    unsigned ArgNo = Arg->getArgNo();
    clang::FunctionDecl *FunDecl = FunctionDecls.at(ArgFun);
    unsigned DeclNumParams = FunDecl->getNumParams();
    revng_assert(NumLLVMParams == DeclNumParams);
    clang::ParmVarDecl *ParamVDecl = FunDecl->getParamDecl(ArgNo);
    QualType Type = ParamVDecl->getType();
    DeclRefExpr *Res = new (ASTCtx)
      DeclRefExpr(ASTCtx, ParamVDecl, false, Type, VK_LValue, {});
    return Res;
  } else {
    revng_abort();
  }
}

Expr *StmtBuilder::getLiteralFromConstant(llvm::Constant *C) {
  if (auto *CD = dyn_cast<ConstantData>(C)) {
    if (auto *CInt = dyn_cast<ConstantInt>(CD)) {
      clang::DeclContext &TUDecl = *ASTCtx.getTranslationUnitDecl();
      QualType LiteralTy = getOrCreateQualType(CInt,
                                               ASTCtx,
                                               TUDecl,
                                               TypeDecls,
                                               FieldDecls);
      const clang::Type *UnderlyingTy = LiteralTy.getTypePtrOrNull();
      revng_assert(UnderlyingTy != nullptr);
      // Desugar stdint.h typedefs
      UnderlyingTy = UnderlyingTy->getUnqualifiedDesugaredType();
      const BuiltinType *BuiltinTy = cast<BuiltinType>(UnderlyingTy);
      switch (BuiltinTy->getKind()) {
      case BuiltinType::Bool: {
        QualType IntT = ASTCtx.IntTy;
        QualType BoolTy = getOrCreateBoolQualType(ASTCtx,
                                                  TypeDecls,
                                                  C->getType());
        uint64_t ConstValue = CInt->getValue().getZExtValue();
        APInt Const = APInt(ASTCtx.getIntWidth(IntT), ConstValue, true);
        Expr *IntLiteral = IntegerLiteral::Create(ASTCtx, Const, IntT, {});
        return createCast(BoolTy, IntLiteral, ASTCtx);
      }
      case BuiltinType::Char_U:
      case BuiltinType::Char_S:
      case BuiltinType::UChar:
      case BuiltinType::SChar: {
        using CharKind = CharacterLiteral::CharacterKind;
        uint64_t ConstValue = CInt->getValue().getZExtValue();
        return new (ASTCtx) CharacterLiteral(static_cast<unsigned>(ConstValue),
                                             CharKind::Ascii,
                                             ASTCtx.CharTy,
                                             {});
      }
      case BuiltinType::UShort: {
        QualType IntT = ASTCtx.UnsignedIntTy;
        QualType ShortT = ASTCtx.UnsignedShortTy;
        uint64_t ConstValue = CInt->getValue().getZExtValue();
        APInt Const = APInt(ASTCtx.getIntWidth(IntT), ConstValue);
        Expr *Literal = IntegerLiteral::Create(ASTCtx, Const, IntT, {});
        return createCast(ShortT, Literal, ASTCtx);
      }
      case BuiltinType::Short: {
        QualType IntT = ASTCtx.IntTy;
        QualType ShortT = ASTCtx.ShortTy;
        uint64_t ConstValue = CInt->getValue().getZExtValue();
        APInt Const = APInt(ASTCtx.getIntWidth(IntT), ConstValue, true);
        Expr *Literal = IntegerLiteral::Create(ASTCtx, Const, IntT, {});
        return createCast(ShortT, Literal, ASTCtx);
      }
      case BuiltinType::UInt:
      case BuiltinType::ULong:
      case BuiltinType::ULongLong: {
        uint64_t ConstValue = CInt->getValue().getZExtValue();
        APInt Const = APInt(ASTCtx.getIntWidth(LiteralTy), ConstValue);
        return IntegerLiteral::Create(ASTCtx, Const, LiteralTy, {});
      }
      case BuiltinType::Int:
      case BuiltinType::Long:
      case BuiltinType::LongLong: {
        uint64_t ConstValue = CInt->getValue().getZExtValue();
        APInt Const = APInt(ASTCtx.getIntWidth(LiteralTy), ConstValue, true);
        return IntegerLiteral::Create(ASTCtx, Const, LiteralTy, {});
      }
      case BuiltinType::UInt128: {
        // With LLVM compiled in debug this asserts whenever ConstValue is
        // larger than 64 bits.
        // We don't use 128 instead of 64 because C hasn't 128 bits integer
        // literals.
        const APInt &OldConst = CInt->getValue();
        unsigned Width = OldConst.getBitWidth();

        // Check that we are not at the boundaries of the representable
        // integers with 64 bit, and in case enforce a full check.
        if (Width <= 64) {
          uint64_t ConstValue = OldConst.getZExtValue();
          APInt Const = APInt(64, ConstValue);
          QualType T = ASTCtx.UnsignedLongLongTy;
          return IntegerLiteral::Create(ASTCtx, Const, T, {});
        } else {
          uint64_t ConstValue = OldConst.getLimitedValue();
          APInt Const = APInt(64, ConstValue);

          // HACK: We actually have values which need 128 bits to be
          // represented, so we disable temporarly the check and simply
          // truncate the value to 64 bit.
          // revng_assert(not Const.isMaxValue());
          QualType T = ASTCtx.UnsignedLongLongTy;
          return IntegerLiteral::Create(ASTCtx, Const, T, {});
        }
      }
      case BuiltinType::Int128: {
        // With LLVM compiled in debug this asserts whenever ConstValue is
        // larger than 64 bits.
        // We don't use 128 instead of 64 because C hasn't 128 bits integer
        // literals.
        const APInt &OldConst = CInt->getValue();
        unsigned Width = OldConst.getBitWidth();

        // Check that we are not at the boundaries of the representable
        // integers with 64 bit, and in case enforce a full check.
        if (Width <= 64) {
          uint64_t ConstValue = OldConst.getZExtValue();
          APInt Const = APInt(64, ConstValue);
          QualType T = ASTCtx.UnsignedLongLongTy;
          return IntegerLiteral::Create(ASTCtx, Const, T, {});
        } else {
          uint64_t ConstValue = OldConst.getLimitedValue();
          APInt Const = APInt(64, ConstValue, true);
          revng_assert(not Const.isMaxSignedValue()
                       and not Const.isMinSignedValue());
          QualType T = ASTCtx.LongLongTy;
          return IntegerLiteral::Create(ASTCtx, Const, T, {});
        }
      }
      case BuiltinType::Dependent:
      case BuiltinType::Overload:
      case BuiltinType::BoundMember:
      case BuiltinType::PseudoObject:
      case BuiltinType::UnknownAny:
      case BuiltinType::BuiltinFn:
      case BuiltinType::ARCUnbridgedCast:
      case BuiltinType::OMPArraySection:
      case BuiltinType::Void:
      case BuiltinType::WChar_U:
      case BuiltinType::WChar_S:
      case BuiltinType::Char8:
      case BuiltinType::Char16:
      case BuiltinType::Char32:
      case BuiltinType::Accum:
      case BuiltinType::ShortAccum:
      case BuiltinType::LongAccum:
      case BuiltinType::UAccum:
      case BuiltinType::UShortAccum:
      case BuiltinType::ULongAccum:
      case BuiltinType::SatAccum:
      case BuiltinType::SatShortAccum:
      case BuiltinType::SatLongAccum:
      case BuiltinType::SatUAccum:
      case BuiltinType::SatUShortAccum:
      case BuiltinType::SatULongAccum:
      case BuiltinType::Fract:
      case BuiltinType::ShortFract:
      case BuiltinType::LongFract:
      case BuiltinType::UFract:
      case BuiltinType::UShortFract:
      case BuiltinType::ULongFract:
      case BuiltinType::SatFract:
      case BuiltinType::SatShortFract:
      case BuiltinType::SatLongFract:
      case BuiltinType::SatUFract:
      case BuiltinType::SatUShortFract:
      case BuiltinType::SatULongFract:
      case BuiltinType::Half:
      case BuiltinType::Float:
      case BuiltinType::Double:
      case BuiltinType::LongDouble:
      case BuiltinType::Float16:
      case BuiltinType::Float128:
      case BuiltinType::NullPtr:
      case BuiltinType::ObjCId:
      case BuiltinType::ObjCClass:
      case BuiltinType::ObjCSel:
      case BuiltinType::OCLSampler:
      case BuiltinType::OCLEvent:
      case BuiltinType::OCLClkEvent:
      case BuiltinType::OCLQueue:
      case BuiltinType::OCLReserveID:
      case BuiltinType::OCLImage1dRO:
      case BuiltinType::OCLImage1dWO:
      case BuiltinType::OCLImage1dRW:
      case BuiltinType::OCLImage1dArrayRO:
      case BuiltinType::OCLImage1dArrayWO:
      case BuiltinType::OCLImage1dArrayRW:
      case BuiltinType::OCLImage1dBufferRO:
      case BuiltinType::OCLImage1dBufferWO:
      case BuiltinType::OCLImage1dBufferRW:
      case BuiltinType::OCLImage2dRO:
      case BuiltinType::OCLImage2dWO:
      case BuiltinType::OCLImage2dRW:
      case BuiltinType::OCLImage2dArrayRO:
      case BuiltinType::OCLImage2dArrayWO:
      case BuiltinType::OCLImage2dArrayRW:
      case BuiltinType::OCLImage2dDepthRO:
      case BuiltinType::OCLImage2dDepthWO:
      case BuiltinType::OCLImage2dDepthRW:
      case BuiltinType::OCLImage2dArrayDepthRO:
      case BuiltinType::OCLImage2dArrayDepthWO:
      case BuiltinType::OCLImage2dArrayDepthRW:
      case BuiltinType::OCLImage2dMSAARO:
      case BuiltinType::OCLImage2dMSAAWO:
      case BuiltinType::OCLImage2dMSAARW:
      case BuiltinType::OCLImage2dArrayMSAARO:
      case BuiltinType::OCLImage2dArrayMSAAWO:
      case BuiltinType::OCLImage2dArrayMSAARW:
      case BuiltinType::OCLImage2dMSAADepthRO:
      case BuiltinType::OCLImage2dMSAADepthWO:
      case BuiltinType::OCLImage2dMSAADepthRW:
      case BuiltinType::OCLImage2dArrayMSAADepthRO:
      case BuiltinType::OCLImage2dArrayMSAADepthWO:
      case BuiltinType::OCLImage2dArrayMSAADepthRW:
      case BuiltinType::OCLImage3dRO:
      case BuiltinType::OCLImage3dWO:
      case BuiltinType::OCLImage3dRW:
      case BuiltinType::OCLIntelSubgroupAVCImePayload:
      case BuiltinType::OCLIntelSubgroupAVCMcePayload:
      case BuiltinType::OCLIntelSubgroupAVCRefPayload:
      case BuiltinType::OCLIntelSubgroupAVCSicPayload:
      case BuiltinType::OCLIntelSubgroupAVCImeResult:
      case BuiltinType::OCLIntelSubgroupAVCMceResult:
      case BuiltinType::OCLIntelSubgroupAVCRefResult:
      case BuiltinType::OCLIntelSubgroupAVCSicResult:
      case BuiltinType::OCLIntelSubgroupAVCImeSingleRefStreamin:
      case BuiltinType::OCLIntelSubgroupAVCImeDualRefStreamin:
      case BuiltinType::OCLIntelSubgroupAVCImeResultSingleRefStreamout:
      case BuiltinType::OCLIntelSubgroupAVCImeResultDualRefStreamout:
      case BuiltinType::SveBool:
      case BuiltinType::SveFloat16:
      case BuiltinType::SveFloat32:
      case BuiltinType::SveFloat64:
      case BuiltinType::SveInt8:
      case BuiltinType::SveInt16:
      case BuiltinType::SveInt32:
      case BuiltinType::SveInt64:
      case BuiltinType::SveUint8:
      case BuiltinType::SveUint16:
      case BuiltinType::SveUint32:
      case BuiltinType::SveUint64:
        revng_abort();
      }
    } else if (isa<ConstantPointerNull>(CD)) {
      QualType UIntPtr = ASTCtx.getUIntPtrType();
      unsigned UIntPtrSize = static_cast<unsigned>(ASTCtx.getTypeSize(UIntPtr));
      return IntegerLiteral::Create(ASTCtx,
                                    APInt::getNullValue(UIntPtrSize),
                                    UIntPtr,
                                    {});
    } else if (isa<UndefValue>(CD)) {
      uint64_t ConstValue = 0;
      APInt Const = APInt(64, ConstValue);
      QualType IntT = ASTCtx.LongTy;
      return IntegerLiteral::Create(ASTCtx, Const, IntT, {});
    }

    revng_abort();
  }
  if (auto *CE = dyn_cast<llvm::ConstantExpr>(C)) {
    Expr *Result = nullptr;
    switch (CE->getOpcode()) {
    case Instruction::Trunc:
    case Instruction::ZExt:
    case Instruction::SExt:
    case Instruction::IntToPtr:
    case Instruction::PtrToInt:
    case Instruction::BitCast: {
      Result = getExprForValue(CE->getOperand(0));
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

} // namespace IR2AST
