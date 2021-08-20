//
// Copyright rev.ng Srls. See LICENSE.md for details.
//

#include <compare>

#include "llvm/ADT/APInt.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Analysis/ScalarEvolutionExpressions.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/Type.h"
#include "llvm/Support/raw_ostream.h"

#include "clang/AST/ASTContext.h"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclGroup.h"
#include "clang/AST/Expr.h"
#include "clang/AST/OperationKinds.h"
#include "clang/AST/RecordLayout.h"
#include "clang/AST/Stmt.h"
#include "clang/AST/Type.h"
#include "clang/Basic/IdentifierTable.h"

#define DISABLE_RECURSIVE_COROUTINES

#include "revng/ADT/RecursiveCoroutine.h"
#include "revng/Support/IRHelpers.h"

#include "revng-c/DataLayoutAnalysis/DLALayouts.h"
#include "revng-c/DataLayoutAnalysis/SCEVBaseAddressExplorer.h"

#include "ASTBuildAnalysis.h"
#include "AddSCEVBarrierPass.h"
#include "DecompilationHelpers.h"
#include "IRASTTypeTranslation.h"
#include "Mangling.h"

static Logger<> ASTBuildLog("ast-builder");

using namespace llvm;
using namespace clang;
using ClangPointerType = clang::PointerType;
using LLVMType = llvm::Type;
using LLVMPointerType = llvm::PointerType;
using TypeDeclOrQualType = DeclCreator::TypeDeclOrQualType;

static std::string dumpToString(const llvm::SCEV *S) {
  std::string Result;
  auto Stream = llvm::raw_string_ostream(Result);
  S->print(Stream);
  return Result;
}

namespace IR2AST {

Expr *StmtBuilder::getParenthesizedExprForValue(const Value *V) {
  Expr *Res = getExprForValue(V);
  if (isa<clang::BinaryOperator>(Res) or isa<ConditionalOperator>(Res))
    Res = new (ASTCtx) ParenExpr({}, {}, Res);
  return Res;
}

Stmt *StmtBuilder::buildStmt(Instruction &I) {
  revng_log(ASTBuildLog, "Build AST for" << dumpToString(&I));

  // If we have type info we try to understand if the current instruction
  // can represents some form of pointer arithmetic that can be translated
  // into a nice access to a field of a struct.
  {
    auto Indent = LoggerIndent(ASTBuildLog);
    if (Stmt *PointerArithmeticStmt = buildPointerArithmeticExpr(I)) {
      revng_log(ASTBuildLog,
                "Built Pointer Arithmetic for: " << dumpToString(&I));
      return PointerArithmeticStmt;
    }
  }

  // If we were not able to emit pointer arithmetic as a nice access to a
  // field struct fallback to normal emission.
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
                                        VK_RValue,
                                        FPOptions());
      return IfStmt::Create(ASTCtx,
                            {},
                            false,
                            nullptr,
                            nullptr,
                            Cond,
                            {},
                            {},
                            GoToThen,
                            {},
                            GoToElse);
    }
  }
  case Instruction::Ret: {
    ReturnInst *Ret = cast<ReturnInst>(&I);
    Value *RetVal = Ret->getReturnValue();

    Expr *ReturnedExpr = nullptr;

    if (auto *ConstRet = dyn_cast_or_null<ConstantStruct>(RetVal)) {
      revng_assert(not VarDecls.count(ConstRet));

      // Create the VarDecl for the local variable
      llvm::Function *TheFunction = Ret->getFunction();
      clang::FunctionDecl &FDecl = Declarator.getFunctionDecl(TheFunction);
      VarDecl *NewVarDecl = createVarDecl(ConstRet, TheFunction, FDecl);
      VarDecls[ConstRet] = NewVarDecl;

      // Create the inizializer
      llvm::SmallVector<clang::Expr *, 8> Initializers;
      for (llvm::Value *V : ConstRet->operands())
        Initializers.push_back(getLiteralFromConstant(cast<llvm::Constant>(V)));
      clang::Expr *InitExpr = new (ASTCtx)
        clang::InitListExpr(ASTCtx, {}, Initializers, {});
      NewVarDecl->setInit(InitExpr);

      ReturnedExpr = getExprForValue(ConstRet);

    } else if (auto *Zero = dyn_cast_or_null<ConstantAggregateZero>(RetVal)) {
      revng_assert(not VarDecls.count(Zero));

      // Create the VarDecl for the local variable
      llvm::Function *TheFunction = Ret->getFunction();
      clang::FunctionDecl &FDecl = Declarator.getFunctionDecl(TheFunction);
      VarDecl *NewVarDecl = createVarDecl(Zero, TheFunction, FDecl);
      VarDecls[Zero] = NewVarDecl;

      // Create the inizializer
      uint64_t ConstValue = 0;
      QualType IntT = ASTCtx.IntTy;
      APInt Const = APInt(ASTCtx.getIntWidth(IntT), ConstValue);
      clang::Expr *ZeroLiteral = IntegerLiteral::Create(ASTCtx,
                                                        Const,
                                                        IntT,
                                                        {});
      clang::Expr *ZeroInit = new (ASTCtx)
        clang::InitListExpr(ASTCtx, {}, { ZeroLiteral }, {});
      NewVarDecl->setInit(ZeroInit);

      ReturnedExpr = getExprForValue(Zero);

    } else {

      ReturnedExpr = RetVal ? getExprForValue(RetVal) : nullptr;
    }
    return ReturnStmt::Create(ASTCtx, {}, ReturnedExpr, nullptr);
  }
  case Instruction::Switch: {
    revng_abort("switch instructions are not supported yet");
    auto *Switch = cast<SwitchInst>(&I);

    Value *Cond = Switch->getCondition();
    Expr *CondE = getExprForValue(Cond);

    SwitchStmt *S = SwitchStmt::Create(ASTCtx, nullptr, nullptr, CondE, {}, {});

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
    revng_assert(I.getParent() == &I.getFunction()->getEntryBlock());
    VarDecl *AllocatedVarDecl = AllocaDecls.at(cast<AllocaInst>(&I));
    QualType AllocatedQualTy = AllocatedVarDecl->getType();
    {
      // TODO: Here we expect allocas to be arrays of bytes.
      // Initially this was the only possibility, because we always created
      // arrays of bytes for allocas. After the introduction of DLA this is no
      // longer the case, so I expecte the following assertion to start failing.
      // When that happens, we'll need to figure out what's the right thing to
      // do here.

      auto *AllocTy = AllocatedQualTy.getTypePtr();
      using clang::ArrayType;
      revng_assert(cast<ArrayType>(AllocTy)->getElementType() == ASTCtx.CharTy);
    }
    // Create an Expr for the address of the first element of the array.
    Expr *AllocatedVarDeclRef = new (ASTCtx) DeclRefExpr(ASTCtx,
                                                         AllocatedVarDecl,
                                                         false,
                                                         AllocatedQualTy,
                                                         VK_LValue,
                                                         {});
    QualType CharPtrTy = ASTCtx.getPointerType(ASTCtx.CharTy);
    CastKind Kind = CastKind::CK_ArrayToPointerDecay;
    Expr *ArrayPtrDecay = ImplicitCastExpr::Create(ASTCtx,
                                                   CharPtrTy,
                                                   Kind,
                                                   AllocatedVarDeclRef,
                                                   nullptr,
                                                   VK_RValue,
                                                   FPOptions());
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
    return clang::UnaryOperator::Create(ASTCtx,
                                        ArraySubscript,
                                        UnaryOperatorKind::UO_AddrOf,
                                        CharPtrTy,
                                        VK_RValue,
                                        OK_Ordinary,
                                        {},
                                        false,
                                        FPOptions());
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
      TypeDeclOrQualType PTy = Declarator.getOrCreateType(Load, ASTCtx, TUDecl);
      QualType PointeeType = DeclCreator::getQualType(PTy);

      QualType QualAddrType = AddrExpr->getType();
      const clang::Type *AddrTy = QualAddrType.getTypePtr();
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

      return clang::UnaryOperator::Create(ASTCtx,
                                          AddrExpr,
                                          UnaryOperatorKind::UO_Deref,
                                          PointeeType,
                                          VK_LValue,
                                          OK_Ordinary,
                                          {},
                                          false,
                                          FPOptions());
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
    return clang::BinaryOperator::Create(ASTCtx,
                                         LHS,
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
    TypeDeclOrQualType LeftQTy = Declarator.getOrCreateType(&I, ASTCtx, TUDecl);
    QualType LHSQualType = DeclCreator::getQualType(LeftQTy);
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
    TypeDeclOrQualType ASTTy = Declarator.getOrCreateType(&I, ASTCtx, TUDecl);

    // Destination type
    QualType ASTType = DeclCreator::getQualType(ASTTy);
    // Result type of the ternary expression.
    QualType TernaryQTy = ASTType;

    {
      QualType TrueQTy = TrueExpr->getType();
      QualType FalseQTy = FalseExpr->getType();

      const clang::Type *TTy = TrueQTy.getTypePtr();
      const clang::Type *FTy = FalseQTy.getTypePtr();

      bool TruePtr = TrueQTy->isPointerType();
      bool FalsePtr = FalseQTy->isPointerType();

      if (not TruePtr and not FalsePtr) {
        // If true and false they are both non-pointes, we do integer promotion,
        // then we will cast to pointer the result of the ternary if necessary.
        int Cmp = ASTCtx.getIntegerTypeOrder(TrueQTy, FalseQTy);
        TernaryQTy = (Cmp > 0) ? TrueQTy : FalseQTy;

      } else {
        // At least true or false are pointers, so we want to promote both sides
        // to pointers.
        if (not TruePtr) {
          // If only false is pointer, we cast true to the same pointer type
          TrueExpr = createCast(FalseQTy, TrueExpr, ASTCtx);
          TernaryQTy = FalseQTy;

        } else if (not FalsePtr) {
          // If only true is pointer, we cast false to the same pointer type
          FalseExpr = createCast(TrueQTy, FalseExpr, ASTCtx);
          TernaryQTy = TrueQTy;

        } else {
          // Both pointers, but they may point to different types.
          auto *UnqualTTy = TTy->getUnqualifiedDesugaredType();
          auto *UnqualFTy = FTy->getUnqualifiedDesugaredType();
          const clang::Type *TernaryTy = ASTType.getTypePtr();

          if (TernaryTy->isPointerType()) {
            // If true and false point to different types, we cast both to the
            // target type of the ternary, if necessary
            if (UnqualTTy != UnqualFTy) {

              if (UnqualTTy != TernaryTy->getUnqualifiedDesugaredType())
                TrueExpr = createCast(TernaryQTy, TrueExpr, ASTCtx);

              if (UnqualFTy != TernaryTy->getUnqualifiedDesugaredType())
                FalseExpr = createCast(TernaryQTy, FalseExpr, ASTCtx);
            }
            TernaryQTy = TrueQTy;
          } else {
            TernaryQTy = ASTCtx.getPointerType(ASTCtx.CharTy);
            TrueExpr = createCast(TernaryQTy, TrueExpr, ASTCtx);
            FalseExpr = createCast(TernaryQTy, FalseExpr, ASTCtx);
          }
        }
      }
    }

    clang::Expr *Ternary = new (ASTCtx) ConditionalOperator(Cond,
                                                            {},
                                                            TrueExpr,
                                                            {},
                                                            FalseExpr,
                                                            TernaryQTy,
                                                            VK_RValue,
                                                            OK_Ordinary);

    if (not ASTCtx.typesAreCompatible(TernaryQTy, ASTType))
      Ternary = createCast(ASTType, Ternary, ASTCtx);
    return Ternary;
  }

  case Instruction::Call: {
    auto *TheCall = cast<CallInst>(&I);

    // Skip llvm.assume() instrinsics
    if (TheCall->getIntrinsicID() == llvm::Intrinsic::assume)
      return nullptr;

    Function *CalleeFun = getCallee(TheCall);

    Expr *CalleeExpr = getExprForValue(CalleeFun);
    revng_log(ASTBuildLog, "GOT!");
    if (ASTBuildLog.isEnabled() and CalleeExpr)
      CalleeExpr->dump();

    size_t NumArgs = CalleeFun->arg_size();
    FunctionDecl &FD = Declarator.getFunctionDecl(CalleeFun);
    size_t NumParms = FD.param_size();
    unsigned NumOps = TheCall->getNumArgOperands();
    bool HasNoParms = NumParms == 0
                      or (NumParms == 1
                          and FD.getParamDecl(0)->getType() == ASTCtx.VoidTy);
    revng_assert(HasNoParms or NumArgs == NumParms);
    const bool IsVariadic = FD.isVariadic();
    if (not FD.isVariadic())
      revng_assert(NumArgs == NumOps);

    auto Args = SmallVector<Expr *, 8>(NumOps, nullptr);
    revng_assert(not(not HasNoParms and IsVariadic));
    if (not HasNoParms) {
      for (unsigned OpId = 0; OpId < NumOps; ++OpId) {
        Value *Operand = TheCall->getOperand(OpId);
        Expr *ArgExpr = getExprForValue(Operand);
        QualType ArgQualTy = ArgExpr->getType();

        ParmVarDecl *ParmDecl = FD.getParamDecl(OpId);
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
    TypeDeclOrQualType RTy = Declarator.getOrCreateType(TheCall->getType(),
                                                        CalleeFun,
                                                        ASTCtx,
                                                        TUDecl);
    QualType ReturnType = DeclCreator::getQualType(RTy);
    return CallExpr::Create(ASTCtx,
                            CalleeExpr,
                            Args,
                            ReturnType,
                            VK_RValue,
                            {},
                            FPOptions());
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
                            {},
                            FPOptions());
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
    revng_assert(AggregateTy->isAggregateType());
    auto *TypeDecl = Declarator.lookupTypeDeclOrNull(AggregateTy);
    auto *StructTypeDecl = cast<clang::RecordDecl>(TypeDecl);
    Expr *StructExpr = getExprForValue(Insert);
    clang::DeclContext &TUDecl = *ASTCtx.getTranslationUnitDecl();
    TypeDeclOrQualType InTy = Declarator.getOrCreateType(Insert->getType(),
                                                         Insert->getFunction(),
                                                         ASTCtx,
                                                         TUDecl);
    revng_assert(not llvm::empty(Insert->indices()));
    unsigned Idx = *Insert->indices().begin();
    FieldDecl *FieldDecl = *std::next(StructTypeDecl->field_begin(), Idx);
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
                                          DeclCreator::getQualType(InTy),
                                          VK_LValue,
                                          OK_Ordinary,
                                          NOUR_None);
    clang::Expr *RHS = getExprForValue(Insert->getInsertedValueOperand());
    BinaryOperatorKind AssignOpKind = BinaryOperatorKind::BO_Assign;
    AdditionalStmts[&I].push_back(clang::BinaryOperator::Create(ASTCtx,
                                                                LHS,
                                                                RHS,
                                                                AssignOpKind,
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
    revng_assert(isa<CallInst>(AggregateOp) or isa<PHINode>(AggregateOp));
    llvm::Type *AggregateTy = AggregateOp->getType();
    revng_assert(AggregateTy->isAggregateType());
    auto *TypeDecl = Declarator.lookupTypeDeclOrNull(AggregateTy);
    auto *StructTypeDecl = cast<clang::RecordDecl>(TypeDecl);
    Expr *StructExpr = getExprForValue(AggregateOp);
    clang::DeclContext &TUDecl = *ASTCtx.getTranslationUnitDecl();
    TypeDeclOrQualType ExtractedTy = Declarator.getOrCreateType(Extract,
                                                                ASTCtx,
                                                                TUDecl);
    revng_assert(not llvm::empty(Extract->indices()));
    unsigned Idx = *Extract->indices().begin();
    auto *ExtractedFDecl = *std::next(StructTypeDecl->field_begin(), Idx);
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
                              DeclCreator::getQualType(ExtractedTy),
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
    TypeDeclOrQualType BoolTy = Declarator.getOrCreateBoolType(ASTCtx);
    SwitchStateVarDecl = VarDecl::Create(ASTCtx,
                                         &FDecl,
                                         {},
                                         {},
                                         &Id,
                                         DeclCreator::getQualType(BoolTy),
                                         nullptr,
                                         StorageClass::SC_None);
    FDecl.addDecl(SwitchStateVarDecl);
  }
  revng_assert(SwitchStateVarDecl != nullptr);
  return SwitchStateVarDecl;
}

clang::Expr *StmtBuilder::getMemberAccessExpr(clang::Expr *BaseExpr,
                                              const LayoutChildInfo &ChildInfo,
                                              bool IsArrow) {
  const auto &[Parent, ChildId] = ChildInfo;

  clang::Expr *Result = nullptr;

  switch (Parent->getKind()) {

  case dla::Layout::LayoutKind::Base: {
    if (llvm::cast<ConstantInt>(ChildId)->getValue().isNullValue())
      Result = BaseExpr;
  } break;

  case dla::Layout::LayoutKind::Array: {
    llvm::Optional<TypeDeclOrQualType> Opt = Declarator.lookupType(Parent);
    revng_assert(Opt.hasValue());
    clang::QualType ArrayQTy = DeclCreator::getQualType(Opt.getValue());
    const clang::Type *ArrayTy = ArrayQTy.getTypePtr();
    clang::QualType ElemTy = ArrayTy->getAsArrayTypeUnsafe()->getElementType();

    // Compute expression for index in the array
    clang::Expr *ArrayIndex = getExprForValue(ChildId);
    revng_assert(ArrayIndex);
    Result = new (ASTCtx) clang::ArraySubscriptExpr(BaseExpr,
                                                    ArrayIndex,
                                                    ElemTy,
                                                    VK_LValue,
                                                    OK_Ordinary,
                                                    {});
  } break;

  case dla::Layout::LayoutKind::Struct:
  case dla::Layout::LayoutKind::Union: {
    llvm::APInt ConstFielId = llvm::cast<ConstantInt>(ChildId)->getValue();
    clang::TypeDecl *Decl = Declarator.lookupTypeDeclOrNull(Parent);
    auto *RecDecl = cast<clang::RecordDecl>(Decl);
    const clang::ASTRecordLayout &RLayout = ASTCtx.getASTRecordLayout(RecDecl);
    revng_assert(ConstFielId.ult(RLayout.getFieldCount()));
    revng_assert(not ConstFielId.isNegative());
    clang::FieldDecl *Field = *std::next(RecDecl->field_begin(),
                                         ConstFielId.getZExtValue());
    revng_assert(Field);

    Result = clang::MemberExpr::CreateImplicit(ASTCtx,
                                               BaseExpr,
                                               IsArrow,
                                               Field,
                                               Field->getType(),
                                               VK_LValue,
                                               OK_Ordinary);
  } break;

  case dla::Layout::LayoutKind::Padding:
  default:
    revng_abort("unexpected dla::Layout");
  }
  return Result;
}

struct NestedChildInfo {
  llvm::SmallVector<StmtBuilder::LayoutChildInfo, 8> ChildInfoVec;
  const llvm::SCEVConstant *ConsumedOffset;
};

RecursiveCoroutine<NestedChildInfo>
getFirstArrayStartingAt(const SCEV *Start,
                        const dla::Layout *BasePointedLayout,
                        llvm::ScalarEvolution *SE,
                        clang::ASTContext &ASTCtx) {

  llvm::LLVMContext &LLVMCtx = Start->getType()->getContext();
  auto *SizeType = llvm::IntegerType::getInt64Ty(LLVMCtx);

  NestedChildInfo Result = {
    {}, llvm::cast<llvm::SCEVConstant>(SE->getConstant(Start->getType(), 0))
  };

  switch (BasePointedLayout->getKind()) {

  case dla::Layout::LayoutKind::Base:
  case dla::Layout::LayoutKind::Padding: {
    // If we reach a base layout or padding, we haven't found any compatible
    // array, so we bail out.
  } break;

  case dla::Layout::LayoutKind::Array: {

    const auto *Array = llvm::cast<dla::ArrayLayout>(BasePointedLayout);
    auto
      *FieldId = llvm::ConstantInt::get(SizeType,
                                        std::numeric_limits<uint64_t>::max());
    Result.ChildInfoVec.push_back({ Array, FieldId });

  } break;

  case dla::Layout::LayoutKind::Struct: {
    auto *Struct = llvm::cast<dla::StructLayout>(BasePointedLayout);
    llvm::APInt Initial = llvm::cast<llvm::SCEVConstant>(Start)->getAPInt();

    auto CumulativeSize = llvm::APInt::getNullValue(Initial.getBitWidth());
    for (auto &Group : llvm::enumerate(Struct->fields())) {

      const dla::Layout *Field = Group.value();
      size_t FieldSize = Field->size();

      // Skip fields that start too late
      if (Initial.uge(CumulativeSize + FieldSize)) {
        CumulativeSize += FieldSize;
        continue;
      }

      const llvm::SCEV *StructStart = SE->getConstant(CumulativeSize);
      const llvm::SCEV *StartInStruct = SE->getMinusSCEV(Start, StructStart);
      NestedChildInfo FieldResult = rc_recur
        getFirstArrayStartingAt(StartInStruct, Field, SE, ASTCtx);

      if (not FieldResult.ChildInfoVec.empty()) {

        auto *FieldId = llvm::ConstantInt::get(SizeType, Group.index());
        Result.ChildInfoVec.push_back({ Struct, FieldId });

        auto &Nested = FieldResult.ChildInfoVec;
        Result.ChildInfoVec.append(Nested.begin(), Nested.end());

        const auto *StartOff = llvm::cast<llvm::SCEVConstant>(StructStart);
        const SCEV *Consumed = SE->getAddExpr(Result.ConsumedOffset, StartOff);
        Consumed = SE->getAddExpr(Consumed, FieldResult.ConsumedOffset);
        Result.ConsumedOffset = llvm::cast<llvm::SCEVConstant>(Consumed);
      }

      break;
    }

  } break;

  case dla::Layout::LayoutKind::Union: {
    auto *Union = llvm::cast<dla::UnionLayout>(BasePointedLayout);
    llvm::APInt Initial = llvm::cast<llvm::SCEVConstant>(Start)->getAPInt();

    llvm::SmallVector<NestedChildInfo, 8> ElemResults;
    const llvm::SCEV *Zero = SE->getConstant(Start->getType(), 0);
    ElemResults.resize(Union->numElements(),
                       { {}, llvm::cast<llvm::SCEVConstant>(Zero) });

    for (auto &Group : llvm::enumerate(Union->elements())) {
      const dla::Layout *Elem = Group.value();
      // Skip elements that are too small
      if (Initial.uge(Elem->size()))
        continue;

      ElemResults[Group.index()] = getFirstArrayStartingAt(Start,
                                                           Elem,
                                                           SE,
                                                           ASTCtx);
    }

    // Choose the first element for which we were able to compute some
    // results.
    // TODO: the child we choose might not be the only one for which we are
    // able to compute a valid result. We should think about policies for
    // better choices in the future.
    for (auto &NonEmptyElemResult : llvm::enumerate(ElemResults)) {

      const auto &[ChildVec, Consumed] = NonEmptyElemResult.value();

      if (ChildVec.empty())
        continue;

      auto *FieldId = llvm::ConstantInt::get(SizeType,
                                             NonEmptyElemResult.index());
      Result.ChildInfoVec.push_back({ Union, FieldId });
      Result.ChildInfoVec.append(ChildVec.begin(), ChildVec.end());

      const llvm::SCEV *NewConsumed = SE->getAddExpr(Result.ConsumedOffset,
                                                     Consumed);
      Result.ConsumedOffset = llvm::cast<llvm::SCEVConstant>(NewConsumed);
      break;
    }

  } break;

  default:
    revng_unreachable("Unknown Layout kind!");
  }

  rc_return Result;
}

RecursiveCoroutine<llvm::SmallVector<StmtBuilder::LayoutChildInfo, 8>>
getNestedFieldIds(const SCEV *Off,
                  const dla::Layout *BasePointedLayout,
                  llvm::ScalarEvolution *SE,
                  clang::ASTContext &ASTCtx) {

  auto Indent = LoggerIndent(ASTBuildLog);

  llvm::LLVMContext &LLVMCtx = Off->getType()->getContext();

  using ResultVec = llvm::SmallVector<StmtBuilder::LayoutChildInfo, 8>;
  ResultVec Result;

  revng_log(ASTBuildLog, dumpToString(Off));

  switch (Off->getSCEVType()) {

  case llvm::scConstant: {
    revng_log(ASTBuildLog, "scConstant");
    auto *ConstOff = llvm::cast<llvm::SCEVConstant>(Off);
    llvm::APInt APOff = ConstOff->getAPInt();

    if (APOff.isNegative()) {
      revng_log(ASTBuildLog, "APOff isNegative: " << dumpToString(Off));
      break;
    }

    if (APOff.uge(BasePointedLayout->size())) {
      revng_log(ASTBuildLog,
                "APOff.uge(size): " << BasePointedLayout->size()
                                    << " <= " << dumpToString(Off));
      break;
    }

    switch (BasePointedLayout->getKind()) {

    case dla::Layout::LayoutKind::Padding: {
      revng_log(ASTBuildLog, "LayoutKind::Padding");
    } break;

    case dla::Layout::LayoutKind::Base: {
      revng_log(ASTBuildLog, "LayoutKind::Base");
      if (APOff.isNullValue()) {
        auto *SizeType = llvm::IntegerType::getInt64Ty(LLVMCtx);
        auto *FieldId = llvm::ConstantInt::get(SizeType, 0);
        Result.push_back({ BasePointedLayout, FieldId });
      }
    } break;

    case dla::Layout::LayoutKind::Array: {
      revng_log(ASTBuildLog, "LayoutKind::Array");
      const auto *Array = llvm::cast<dla::ArrayLayout>(BasePointedLayout);
      llvm::APInt Remainder;
      llvm::APInt Quotient;
      llvm::APInt ElemeSize(APOff.getBitWidth(), Array->getElem()->size());
      llvm::APInt::udivrem(APOff, ElemeSize, Quotient, Remainder);

      if (Remainder.isNullValue()) {
        // Found
        auto *FieldId = llvm::ConstantInt::get(ConstOff->getType(), Quotient);
        Result.push_back({ Array, FieldId });
      } else {
        const SCEV *OffInElem = SE->getConstant(Remainder);
        ResultVec ChildResult = rc_recur getNestedFieldIds(OffInElem,
                                                           Array->getElem(),
                                                           SE,
                                                           ASTCtx);
        if (not ChildResult.empty()) {
          auto *FieldId = llvm::ConstantInt::get(ConstOff->getType(), Quotient);
          Result.push_back({ Array, FieldId });
          Result.append(ChildResult.begin(), ChildResult.end());
        }
      }

    } break;

    case dla::Layout::LayoutKind::Struct: {
      revng_log(ASTBuildLog, "LayoutKind::Struct");
      auto *Struct = llvm::cast<dla::StructLayout>(BasePointedLayout);

      auto CumulativeSize = llvm::APInt::getNullValue(APOff.getBitWidth());
      for (auto &Group : llvm::enumerate(Struct->fields())) {

        const dla::Layout *Field = Group.value();
        size_t FieldSize = Field->size();

        if (APOff.uge(CumulativeSize + FieldSize)) {
          CumulativeSize += FieldSize;
          continue;
        }

        const llvm::SCEV *StructStart = SE->getConstant(CumulativeSize);
        const llvm::SCEV *OffInStruct = SE->getMinusSCEV(Off, StructStart);
        ResultVec FieldResult = rc_recur getNestedFieldIds(OffInStruct,
                                                           Field,
                                                           SE,
                                                           ASTCtx);

        if (not FieldResult.empty()) {
          auto *SizeType = llvm::IntegerType::getInt64Ty(LLVMCtx);
          auto *FieldId = llvm::ConstantInt::get(SizeType, Group.index());
          Result.push_back({ Struct, FieldId });
          Result.append(FieldResult.begin(), FieldResult.end());
        }

        break;
      }

    } break;

    case dla::Layout::LayoutKind::Union: {
      revng_log(ASTBuildLog, "LayoutKind::Union");
      auto *Union = llvm::cast<dla::UnionLayout>(BasePointedLayout);

      auto ElemResults = llvm::SmallVector<ResultVec, 8>(Union->numElements(),
                                                         {});

      for (auto &Group : llvm::enumerate(Union->elements())) {

        const dla::Layout *FieldLayout = Group.value();
        if (APOff.uge(FieldLayout->size()))
          continue;

        const llvm::SCEV *Zero = SE->getZero(Off->getType());
        ElemResults[Group.index()] = rc_recur getNestedFieldIds(Zero,
                                                                FieldLayout,
                                                                SE,
                                                                ASTCtx);
      }

      // Choose the first element for which we were able to compute some
      // results.
      // TODO: the child we choose might not be the only one for which we are
      // able to compute a valid result. We should think about policies for
      // better choices in the future.
      for (auto &NonEmptyElemResult : llvm::enumerate(ElemResults)) {

        if (NonEmptyElemResult.value().empty())
          continue;

        auto *SizeType = llvm::IntegerType::getInt64Ty(LLVMCtx);
        auto *FieldId = llvm::ConstantInt::get(SizeType,
                                               NonEmptyElemResult.index());
        Result.push_back({ Union, FieldId });
        Result.append(NonEmptyElemResult.value().begin(),
                      NonEmptyElemResult.value().end());
        break;
      }

    } break;

    default:
      revng_unreachable("Unknown Layout kind!");
    }

  } break;

  case llvm::scAddRecExpr: {
    revng_log(ASTBuildLog, "scAddRecExpr");

    // TODO: by breaking out here, we are explicitly disabling the emission of
    // array accesses in C. This is necessary because at the moment they are
    // stil broken and require to restructure loops in a well-formed shape to
    // fix their emission.
    // However, we still want to keep emitting all the rest that does not
    // have anything to do with arrays, because the rest is supposed to be
    // correct already.
    break;

    // Setup a vector of nested addrecs.
    // We expect the first (most external one) to have
    // a larger step.
    const llvm::SCEVConstant *RecStart = nullptr;
    llvm::SmallVector<const llvm::SCEVAddRecExpr *, 16> NestedAddRecs;
    {

      const auto *AddRecOff = llvm::cast<llvm::SCEVAddRecExpr>(Off);
      revng_assert(AddRecOff->isAffine());

      while (AddRecOff) {
        // We are only able to process nested recurring expressions for which
        // all the increments are constants, and non-negative.
        // For all the others we bail out.
        auto *Incr = dyn_cast<SCEVConstant>(AddRecOff->getStepRecurrence(*SE));
        if (not Incr or Incr->getAPInt().isNegative())
          break;

        if (not NestedAddRecs.empty()) {
          const auto *OuterAddRec = NestedAddRecs.back();
          const auto *OuterAddRecStep = OuterAddRec->getStepRecurrence(*SE);
          auto *OuterIncr = cast<SCEVConstant>(OuterAddRecStep);
          if (OuterIncr->getAPInt().ult(Incr->getAPInt())) {
            // This may happens in nasty functions like memchr.
            // In that case, we stop here and we don't go much further into the
            // nesting.
            break;
          }
        }

        NestedAddRecs.push_back(AddRecOff);
        RecStart = dyn_cast<SCEVConstant>(AddRecOff->getStart());
        AddRecOff = dyn_cast<SCEVAddRecExpr>(AddRecOff->getStart());
      }

      // If we haven't reached the bottom of the nested recurring expression, or
      // we have but the start is not a constant, we cannot do anything, so we
      // just bail out.
      if (not RecStart) {
        NestedAddRecs.clear();
      } else {
        revng_assert(not AddRecOff);
        revng_assert(not NestedAddRecs.empty());
      }
    }

    revng_assert(NestedAddRecs.empty() == (RecStart == nullptr));

    if (not RecStart) {
      revng_assert(Result.empty());
      break;
    }

    ResultVec PartialResults; // should be emptied on fail
    for (const llvm::SCEVAddRecExpr *AddRecOff : NestedAddRecs) {

      const llvm::Loop *Loop = AddRecOff->getLoop();
      llvm::PHINode *IndVar = Loop->getInductionVariable(*SE);
      if (not IndVar) {
        PartialResults.clear();
        break;
      }

      NestedChildInfo
        ResultUntilArray = getFirstArrayStartingAt(RecStart,
                                                   BasePointedLayout,
                                                   SE,
                                                   ASTCtx);

      const auto &[UntilArrayVec, Consumed] = ResultUntilArray;
      revng_assert(UntilArrayVec.empty()
                   or isa<dla::ArrayLayout>(UntilArrayVec.back().Parent));

      // On this AddRecOff we didn't find an array where we expected it, so we
      // have to bail out.
      if (UntilArrayVec.empty()) {
        PartialResults.clear();
        break;
      }

      // Here we expect to handle nested recurring expressions with constant
      // positive increments.
      auto *Incr = cast<SCEVConstant>(AddRecOff->getStepRecurrence(*SE));
      revng_assert(not Incr->getAPInt().isNegative());

      // This is the array that we have found.
      const auto *A = cast<dla::ArrayLayout>(UntilArrayVec.back().Parent);

      // We expect to find an array whose element has the same size of the loop
      // increment, otherwise something is wrong and we have to bail out.
      if (Incr->getAPInt() != A->getElem()->size()) {
        PartialResults.clear();
        break;
      }

      PartialResults.append(UntilArrayVec.begin(), UntilArrayVec.end());
      PartialResults.back().ChildId = IndVar;

      // At the next iteration, BasePointedLayout starts from the elemen of this
      // array.
      BasePointedLayout = A->getElem();
      const llvm::SCEV *RemainingStart = SE->getMinusSCEV(RecStart, Consumed);
      RecStart = llvm::cast<llvm::SCEVConstant>(RemainingStart);
    }

    if (not PartialResults.empty()) {
      Result.append(PartialResults.begin(), PartialResults.end());
    } else {
      revng_assert(Result.empty());
    }

  } break;

  case llvm::scUnknown:
  case llvm::scUDivExpr:
  case llvm::scUMaxExpr:
  case llvm::scSMaxExpr:
  case llvm::scUMinExpr:
  case llvm::scSMinExpr:
  case llvm::scTruncate:
  case llvm::scSignExtend:
  case llvm::scZeroExtend:
  case llvm::scMulExpr:
  case llvm::scAddExpr: {
    // Bail out in these cases
    revng_log(ASTBuildLog, "Unhandled offset SCEV kind");
  } break;

  case llvm::scCouldNotCompute:
  default:
    revng_unreachable("Unknown SCEV kind!");
  }

  rc_return Result;
}

clang::Expr *StmtBuilder::buildPointerArithmeticExpr(llvm::Instruction &I) {

  // If we have no ValueLayouts, the DLA did not run, so we do nothing.
  if (not ValueLayouts)
    return nullptr;

  revng_assert(SE);

  // If I is not SCEVable, we can't work with SCEVs, so we can't to anything.
  if (not SE->isSCEVable(I.getType())) {
    revng_log(ASTBuildLog, "NOT SCEVABLE");
    return nullptr;
  }

  const SCEV *ISCEV = SE->getSCEV(&I);
  // Compute the base address of the Instruction SCEV
  auto Bases = SCEVBaseAddressExplorer().findBases(SE, ISCEV, {});

  // We expect no bases or at most one base address. If we get more than
  // one possible candidate base address for ISCEV we haven't decided
  // which type to emit yet, so this is not handled.
  if (Bases.size() > 1) {
    revng_log(ASTBuildLog, "MANY BASES");
    return nullptr;
  }

  // It was impossible to find a valid base address for ISCEV.
  // This means that we can still try to interpret ISCEV as base
  // address of itself. Otherwise, the first base is considered the address.
  const SCEV *Base = Bases.empty() ? ISCEV : *Bases.begin();
  revng_assert(Base);

  // We assume that if we find a Base SCEV, its is a SCEVUnknown, and we can get
  // its Value, which is the associated base address in the LLMV IR.
  // If it's not, we can't do anything for now.
  // However, this is a potential spot to detect loop induction variables in the
  // future.
  if (not isa<llvm::SCEVUnknown>(Base)) {
    revng_log(ASTBuildLog, "UNKNOWN BASE");
    return nullptr;
  }

  // If Base == ISCEV it means that we have no pointer arithmetic to do at all,
  // so we can just bail out.
  if (ISCEV == Base) {
    revng_log(ASTBuildLog, "BASE OF ITSELF");
    return nullptr;
  }

  const auto *BaseValue = cast<llvm::SCEVUnknown>(Base)->getValue();

  // Unwrap calls to revng_scev_barrier_*
  // TODO: calls to revng_scev_barrier_* should eventually be removed after
  // using them and before actually generating C code for them.
  if (auto *Call = dyn_cast<CallInst>(BaseValue)) {
    if (Call->getType()->isIntOrPtrTy()) {

      const llvm::Type *BarrierTy = Call->getType();
      const llvm::Function *SCEVBarrier = Call->getCalledFunction();
      const std::string BarrierName = makeSCEVBarrierName(BarrierTy);

      if (SCEVBarrier->getName().str() == BarrierName) {
        revng_assert(SCEVBarrier->arg_size() == 1);
        BaseValue = Call->getArgOperand(0);
      }
    }
  }

  // Try to obtain the DLA type of BaseValue
  auto *TUDecl = ASTCtx.getTranslationUnitDecl();
  llvm::Optional<TypeDeclOrQualType>
    DLAType = Declarator.getOrCreateDLAType(BaseValue, ASTCtx, *TUDecl);

  // If we can't obtain the DLA type of BaseValue, we have nothing to work on to
  // properly emit the address arithmetic expression, so we bail out.
  if (not DLAType.hasValue()) {
    revng_log(ASTBuildLog, "DLA TYPE NOT FOUND");
    return nullptr;
  }

  QualType DLAQualTy = DeclCreator::getQualType(DLAType.getValue());
  // We only accept DLA Types that are pointers to structs.
  if (not DLAQualTy.getTypePtr()->isPointerType()) {
    revng_log(ASTBuildLog, "DLA TYPE IS NOT A POINTER");
    return nullptr;
  }

  auto BasePointedLayouts = Declarator.getPointedLayouts(BaseValue);
  // TODO: This assertion is eventually bound to fail whenever BaseValue has a
  // struct type. For now we don't handle that case, but we will need to do it.
  // This is just a hard reminder that we have to handle that case.
  revng_assert(BasePointedLayouts.size() == 1);
  const dla::Layout *BasePointedLayout = BasePointedLayouts.front();
  revng_assert(BasePointedLayout);

  // Compute SCEV for (- Base), and make sure (- Base) has the same size as
  // ISCEV.
  const SCEV *MinusBase = nullptr;
  auto BaseSize = SE->getTypeSizeInBits(Base->getType());
  auto ISCEVSize = SE->getTypeSizeInBits(ISCEV->getType());
  std::strong_ordering Cmp = BaseSize <=> ISCEVSize;
  if (Cmp < 0) {

    // If Base is narrower, zero extend it and negate it.
    // Leave ISCEV like it is.
    const SCEV *ExtBase = SE->getZeroExtendExpr(Base, ISCEV->getType());
    MinusBase = SE->getNegativeSCEV(ExtBase);

  } else if (Cmp > 0) {

    // If Base is wider, just negate it, and zero extend ISCEV.
    MinusBase = SE->getNegativeSCEV(Base);
    ISCEV = SE->getZeroExtendExpr(ISCEV, Base->getType());

  } else { // Otherwise just negate Base.
    MinusBase = SE->getNegativeSCEV(Base);
  }
  revng_assert(MinusBase);

  // Off = I - Base
  const SCEV *Off = SE->getAddExpr(ISCEV, MinusBase);

  llvm::SmallVector<LayoutChildInfo, 8>
    NestedFields = getNestedFieldIds(Off, BasePointedLayout, SE, ASTCtx);

  if (NestedFields.empty()) {
    revng_log(ASTBuildLog, "CANNOT FIND PROPER NESTING");
    return nullptr;
  }

  clang::Expr *Result = getExprForValue(BaseValue);

  // The first MemberExpr always has an arrow (BaseValue->field1) because
  // BaseValue is a pointer.
  Result = getMemberAccessExpr(Result,
                               NestedFields.front(),
                               /* IsArrow */ true);
  if (Result == nullptr) {
    revng_log(ASTBuildLog, "CANNOT BUILD EXPRESSION");
    return nullptr;
  }

  // Create all the field past the first, if any.
  // All the subsequent, if present, have a dot (field1.field2).
  for (const LayoutChildInfo &ChildInfo : llvm::drop_begin(NestedFields, 1)) {
    if (llvm::isa<dla::BaseLayout>(ChildInfo.Parent)) {
      revng_assert(&ChildInfo == &NestedFields.back());
      continue;
    }

    Result = getMemberAccessExpr(Result, ChildInfo, /* IsArrow */ false);
    if (Result == nullptr) {
      revng_log(ASTBuildLog, "CANNOT BUILD NESTED EXPRESSION");
      return nullptr;
    }
  }

  // Wrap all the pointer arithmetic inside an AddrOf expression, to prevent the
  // computed expression to have side effects.
  // In this way we obtain &BaseValue->field1.field2.fieldn;
  clang::QualType AddressType = ASTCtx.getPointerType(Result->getType());
  Result = clang::UnaryOperator::Create(ASTCtx,
                                        Result,
                                        UnaryOperatorKind::UO_AddrOf,
                                        AddressType,
                                        VK_RValue,
                                        OK_Ordinary,
                                        {},
                                        false,
                                        FPOptions());
  return Result;
}

void StmtBuilder::createAST(llvm::Function &F, clang::FunctionDecl &FDecl) {
  revng_log(ASTBuildLog,
            "Building AST for Instructions in Function " << F.getName());

  revng_assert(not ValueLayouts or SE);

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
      // Skip calls to `revng_scev_barrier_*`
      // TODO: calls to revng_scev_barrier_* should eventually be removed after
      // using them and before actually generating C code for them.
      if (auto *Call = dyn_cast<CallInst>(&I)) {
        if (Call->getType()->isIntOrPtrTy()) {

          const llvm::Type *BarrierTy = Call->getType();
          const std::string BarrierName = makeSCEVBarrierName(BarrierTy);
          const llvm::Function *SCEVBarrier = Call->getCalledFunction();

          if (SCEVBarrier->getName().str() == BarrierName) {
            revng_assert(SCEVBarrier->arg_size() == 1);
            InstrStmts[&I] = getExprForValue(Call->getArgOperand(0));

            continue;
          }
        }
      }

      // We don't build clang's AST expressions for PHINodes nor for
      // BranchInsts and SwitchInsts.

      // For BranchInsts, we don't create AST right now, because the emission of
      // control flow statements in C is driven by the ASTTree
      if (isa<BranchInst>(&I))
        continue;

      // For SwitchInsts, we don't create AST right now, because the emission of
      // control flow statements in C is driven by the ASTTree
      if (isa<SwitchInst>(&I))
        continue;

      // PHINodes are not expanded into expressions because they expand in a
      // local variable, that is assigned multiple times for all the incoming
      // Values of the PHINode.
      // Each PHINode has an associated VarDecl
      if (isa<PHINode>(&I)) {
        revng_assert(VarDecls.count(&I) == 0);
        VarDecl *NewVarDecl = createVarDecl(&I, FDecl);
        VarDecls[&I] = NewVarDecl;
        continue;
      }

      // Declare a special local variable for those instructions that need it to
      // build the expression.
      // Examples are AllocaInst and InsertValueInst.
      auto ToSerializeIt = ToSerialize.find(&I);
      auto ToSerializeEnd = ToSerialize.end();
      if (ToSerializeIt != ToSerializeEnd
          and ToSerializeIt->second.isSet(NeedsLocalVarToComputeExpr)) {
        if (auto *Alloca = dyn_cast<AllocaInst>(&I)) {
          // TODO: for now we ignore the alignment of the alloca. This might
          // turn out not to be safe later, because it does not take into
          // account the alignment of future accesses in the `Alloca`ted space.
          // If the code is then recompiled for an architecture that does not
          // support unaligned access this may cause crashes.
          revng_assert(Alloca->isStaticAlloca());
          revng_assert(AllocaDecls.count(Alloca) == 0);
          VarDecl *NewAllocaDecl = createVarDecl(Alloca, FDecl);
          AllocaDecls[Alloca] = NewAllocaDecl;
        } else if (auto *Insert = dyn_cast<InsertValueInst>(&I)) {
          revng_assert(VarDecls.count(Insert) == 0);
          VarDecl *NewVarDecl = createVarDecl(Insert, FDecl);
          VarDecls[Insert] = NewVarDecl;

          // Setup the initial value for the NewVarDecl.
          // This value will be emitted as an intialization.
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
          } else if (isa<InsertValueInst>(AggregateOp)
                     or isa<UndefValue>(AggregateOp)) {
            // If the InsertValueInst is inserting something inside an undef
            // aggregate, or inside a struct coming from another InsertValue, we
            // simply don't initialize it.
            // Given that the initialization can be dynamic, we just leave its
            // handling to the actual emission of the assignments that happens
            // later, in the call to buildStmt.
          } else {
            revng_unreachable();
          }
        } else {
          revng_unreachable();
        }
      }

      Stmt *NewStmt = buildStmt(I);

      // If we didn't emit anything, just skip the rest.
      if (not NewStmt)
        continue;

      InstrStmts[&I] = NewStmt;

      // Build the local variable for all the other instructions that need it.
      if (ToSerializeIt != ToSerializeEnd) {
        const SerializationFlags Flags = ToSerializeIt->second;
        revng_assert(not Flags.isSet(NeedsManyStatements)
                     or Flags.isSet(NeedsLocalVarToComputeExpr));

        if (SerializationFlags::needsVarDecl(Flags)
            and not Flags.isSet(NeedsLocalVarToComputeExpr)) {

          revng_assert(VarDecls.count(&I) == 0);
          VarDecl *NewVarDecl = createVarDecl(&I, FDecl);
          VarDecls[&I] = NewVarDecl;
        }
      }
    }
  }
}

VarDecl *
StmtBuilder::createVarDecl(const Instruction *I, clang::FunctionDecl &FDecl) {
  clang::DeclContext &TUDecl = *ASTCtx.getTranslationUnitDecl();

  TypeDeclOrQualType ASTType;

  llvm::Optional<TypeDeclOrQualType>
    DLAType = Declarator.getOrCreateDLAType(I, ASTCtx, TUDecl);

  if (DLAType.hasValue()) {

    ASTType = DLAType.getValue();

  } else {

    if (const auto *Alloca = dyn_cast<AllocaInst>(I)) {
      // First, create a VarDecl, for an array of char to place in the
      // BasicBlock where the AllocaInst is
      const DataLayout &DL = I->getModule()->getDataLayout();
      uint64_t AllocaSize = *Alloca->getAllocationSizeInBits(DL);
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
      ASTType = ArrayTy;
    } else if (const auto *Call = dyn_cast<llvm::CallInst>(I)) {
      ASTType = Declarator.getOrCreateType(Call->getType(),
                                           Call->getCalledFunction(),
                                           ASTCtx,
                                           TUDecl);
    } else if (const auto *Insert = dyn_cast<llvm::InsertValueInst>(I)) {
      ASTType = Declarator.getOrCreateType(Insert->getType(),
                                           Insert->getFunction(),
                                           ASTCtx,
                                           TUDecl);
    } else {
      clang::DeclContext &TUDecl = *ASTCtx.getTranslationUnitDecl();
      ASTType = Declarator.getOrCreateType(I, ASTCtx, TUDecl);
    }
  }

  clang::QualType ASTQualType = DeclCreator::getQualType(ASTType);
  revng_assert(not ASTQualType.isNull());
  const std::string VarName = I->hasName() ?
                                I->getName().str() :
                                (std::string("var_") + std::to_string(NVar++));
  IdentifierInfo &Id = ASTCtx.Idents.get(makeCIdentifier(VarName));
  VarDecl *NewVarDecl = VarDecl::Create(ASTCtx,
                                        &FDecl,
                                        {},
                                        {},
                                        &Id,
                                        ASTQualType,
                                        nullptr,
                                        StorageClass::SC_None);
  // Add the NewVarDecl to the function declaration context, so that clang's
  // AST printer will print the variable declaration.
  FDecl.addDecl(NewVarDecl);
  return NewVarDecl;
}

VarDecl *StmtBuilder::createVarDecl(Constant *C,
                                    Value *NamingVal,
                                    clang::FunctionDecl &FDecl) {
  clang::DeclContext &TUDecl = *ASTCtx.getTranslationUnitDecl();
  TypeDeclOrQualType ASTType = clang::QualType();
  if (auto *ZeroAggregate = dyn_cast<llvm::ConstantAggregateZero>(C)) {
    ASTType = Declarator.getOrCreateType(ZeroAggregate->getType(),
                                         NamingVal,
                                         ASTCtx,
                                         TUDecl);
  } else if (auto *ConstStruct = dyn_cast<llvm::ConstantStruct>(C)) {
    ASTType = Declarator.getOrCreateType(ConstStruct->getType(),
                                         NamingVal,
                                         ASTCtx,
                                         TUDecl);
  } else {
    revng_abort("trying to create VarDecl for unexpected constant");
  }

  QualType ASTQualType = DeclCreator::getQualType(ASTType);
  revng_assert(not ASTQualType.isNull());
  const std::string VarName = C->hasName() ?
                                C->getName().str() :
                                (std::string("var_") + std::to_string(NVar++));
  IdentifierInfo &Id = ASTCtx.Idents.get(makeCIdentifier(VarName));
  VarDecl *NewVarDecl = VarDecl::Create(ASTCtx,
                                        &FDecl,
                                        {},
                                        {},
                                        &Id,
                                        ASTQualType,
                                        nullptr,
                                        StorageClass::SC_None);
  FDecl.addDecl(NewVarDecl);
  return NewVarDecl;
}

static clang::BinaryOperatorKind
getClangBinaryOpKind(const Instruction &I,
                     const clang::Type *LHSTy,
                     const clang::Type *RHSTy) {
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
    if (LHSTy->isBooleanType() and RHSTy->isBooleanType())
      Res = clang::BinaryOperatorKind::BO_LAnd;
    else
      Res = clang::BinaryOperatorKind::BO_And;
  } break;
  case Instruction::Or: {
    if (LHSTy->isBooleanType() and RHSTy->isBooleanType())
      Res = clang::BinaryOperatorKind::BO_LOr;
    else
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

static std::pair<Expr *, Expr *> getCastedBinaryOperands(ASTContext &ASTCtx,
                                                         const Instruction &I,
                                                         Expr *LHS,
                                                         Expr *RHS) {

  const clang::Type *LHSTy = LHS->getType().getTypePtr();
  const clang::Type *RHSTy = RHS->getType().getTypePtr();
  revng_assert((LHSTy->isPointerType() or LHSTy->isIntegerType())
               and (RHSTy->isPointerType() or RHSTy->isIntegerType()));

  unsigned OpCode = I.getOpcode();

  if (LHSTy->isPointerType() or RHSTy->isPointerType()) {

    switch (OpCode) {
    case Instruction::ICmp: {
      // If it's an equality comparison and only one of the operands is a
      // pointer, promote them both to pointer.
      if (LHSTy->isPointerType() and not RHSTy->isPointerType()) {
        RHS = ImplicitCastExpr::Create(ASTCtx,
                                       LHS->getType(),
                                       CastKind::CK_IntegralToPointer,
                                       RHS,
                                       nullptr,
                                       VK_RValue,
                                       FPOptions());
        RHSTy = LHSTy;
      } else if (not LHSTy->isPointerType() and RHSTy->isPointerType()) {
        LHS = ImplicitCastExpr::Create(ASTCtx,
                                       RHS->getType(),
                                       CastKind::CK_IntegralToPointer,
                                       LHS,
                                       nullptr,
                                       VK_RValue,
                                       FPOptions());
        LHSTy = RHSTy;
      }
    } break;

    case Instruction::Add: {
      // If it's an additive expression (see C99 standard 6.5.6), we can leave
      // them like they are. The pointer remains a pointer, and the integer
      // represents an offset.
      // However, in case of sums, only one of the operands can be of pointer
      // type.
      if (LHSTy->isPointerType() and RHSTy->isPointerType()) {

        QualType IntPtrTy = ASTCtx.getUIntPtrType();
        TypeSourceInfo *TI = ASTCtx.CreateTypeSourceInfo(IntPtrTy);
        LHS = CStyleCastExpr::Create(ASTCtx,
                                     IntPtrTy,
                                     VK_RValue,
                                     CK_PointerToIntegral,
                                     LHS,
                                     nullptr,
                                     FPOptions(),
                                     TI,
                                     {},
                                     {});
        LHSTy = IntPtrTy.getTypePtr();

        RHS = CStyleCastExpr::Create(ASTCtx,
                                     IntPtrTy,
                                     VK_RValue,
                                     CK_PointerToIntegral,
                                     RHS,
                                     nullptr,
                                     FPOptions(),
                                     TI,
                                     {},
                                     {});
        RHSTy = IntPtrTy.getTypePtr();
      }
    } break;

    case Instruction::Sub: {
      // If it's an additive expression (see C99 standard 6.5.6), we can leave
      // them like they are. The pointer remains a pointer, and the integer
      // represents an offset.
      break;
    }

    case Instruction::AShr:
    case Instruction::LShr:
    case Instruction::Shl:
    case Instruction::SDiv:
    case Instruction::UDiv:
    case Instruction::SRem:
    case Instruction::URem:
    case Instruction::Mul:
    case Instruction::And:
    case Instruction::Or:
    case Instruction::Xor: {

      // This is supposed to never happen, but it does,
      // due to DLA, so we have to handle it.

      QualType IntPtrTy = ASTCtx.getUIntPtrType();
      TypeSourceInfo *TI = ASTCtx.CreateTypeSourceInfo(IntPtrTy);

      if (LHSTy->isPointerType()) {
        LHS = CStyleCastExpr::Create(ASTCtx,
                                     IntPtrTy,
                                     VK_RValue,
                                     CK_PointerToIntegral,
                                     LHS,
                                     nullptr,
                                     FPOptions(),
                                     TI,
                                     {},
                                     {});
        LHSTy = IntPtrTy.getTypePtr();
      }

      if (RHSTy->isPointerType()) {
        RHS = CStyleCastExpr::Create(ASTCtx,
                                     IntPtrTy,
                                     VK_RValue,
                                     CK_PointerToIntegral,
                                     RHS,
                                     nullptr,
                                     FPOptions(),
                                     TI,
                                     {},
                                     {});
        RHSTy = IntPtrTy.getTypePtr();
      }

    } break;

    default:
      revng_abort();
    }
  }

  uint64_t LHSSize = ASTCtx.getTypeSize(LHSTy);
  uint64_t RHSSize = ASTCtx.getTypeSize(RHSTy);
  uint64_t MaxSize = std::max(LHSSize, RHSSize);
  unsigned Size = static_cast<unsigned>(MaxSize);

  QualType SignedTy = ASTCtx.getIntTypeForBitwidth(Size, /* Signed */ true);

  std::pair<Expr *, Expr *> Res = std::make_pair(LHS, RHS);
  switch (OpCode) {
    // These instructions have unsigned semantics in llvm IR.
    // We emit unsigned integers by default, so these operations do not need
    // any cast to preserve the semantics in C.

  case Instruction::Add:
  case Instruction::Sub:
  case Instruction::And:
  case Instruction::Or:
  case Instruction::Xor:
    // This set of instructions (described in paragraphs 6.5.6 'Additive
    // operators', paragraph 6.5.10 'Bitwise AND operator', paragraph 6.5.11
    // 'Bitwise exclusive OR operator', and paragraph 6.5.12 'Bitwise
    // inclusive OR operator' of the C11 standard) may have a large unsigned
    // integer literal as one or both operands. In those cases, it is
    // beneficial for the readability of the generate C code to substitute
    // such large unsigned integer literal with negative signed integer
    // literal. This enables printing idiomatic expressions such as 'X - 1'
    // instead of 'X + 0xFFFFFFFFFFFFFFFF'.

    if (auto *RHSLiteral = dyn_cast<clang::IntegerLiteral>(RHS)) {
      llvm::APInt RHSVal = RHSLiteral->getValue();
      revng_assert(RHSVal.getBitWidth() == RHSSize);
      if (RHSVal.isNegative()) {
        QualType SIntT = ASTCtx.getIntTypeForBitwidth(RHSVal.getBitWidth(),
                                                      /*signed*/ true);
        auto NegRHS = IntegerLiteral::Create(ASTCtx, RHSVal, SIntT, {});
        Res.second = new (ASTCtx) ParenExpr({}, {}, NegRHS);
      }
    }

    [[fallthrough]];

  case Instruction::Shl:
  case Instruction::LShr:
    // Shifts are undefined behavior if the RHS is negative (see
    // paragraph 6.5.7 of the C11 standard: 'Bitwise shift operators'), so we
    // don't try to promote big unsigned integer literals at constants RHS to
    // negative signed integer literals.
    //
    if (auto *LHSLiteral = dyn_cast<clang::IntegerLiteral>(LHS)) {
      llvm::APInt LHSVal = LHSLiteral->getValue();
      revng_assert(LHSVal.getBitWidth() == LHSSize);
      if (LHSVal.isNegative()) {
        QualType SIntT = ASTCtx.getIntTypeForBitwidth(LHSVal.getBitWidth(),
                                                      /*signed*/ true);
        auto NegLHS = IntegerLiteral::Create(ASTCtx, LHSVal, SIntT, {});
        Res.second = new (ASTCtx) ParenExpr({}, {}, NegLHS);
      }
    }

    [[fallthrough]];

  case Instruction::Mul:
  case Instruction::UDiv:
  case Instruction::URem:
    // For multiplication, division, and reminder (paragraph 6.5.5 of the C11
    // standard: 'Multiplicative operators'), we could in principle promote
    // big positive unsigned integer literals to negative signed literals, but
    // the consequence on the sign of the result are not clear to me now, so I
    // just leave them like they are for now.
    {}
    break;

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
                                   VK_RValue,
                                   FPOptions());

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
                                   VK_RValue,
                                   FPOptions());

  std::tie(LHS, RHS) = getCastedBinaryOperands(ASTCtx, I, LHS, RHS);

  auto BinOpKind = getClangBinaryOpKind(I,
                                        LHS->getType().getTypePtr(),
                                        RHS->getType().getTypePtr());
  unsigned OpCode = I.getOpcode();
  clang::QualType ResTy = (OpCode == Instruction::ICmp) ? ASTCtx.BoolTy :
                                                          LHS->getType();
  Expr *Res = clang::BinaryOperator::Create(ASTCtx,
                                            LHS,
                                            RHS,
                                            BinOpKind,
                                            ResTy,
                                            VK_RValue,
                                            OK_Ordinary,
                                            {},
                                            FPOptions());

  switch (OpCode) {
  case Instruction::SDiv:
  case Instruction::SRem:
  case Instruction::AShr: {
    clang::DeclContext &TUDecl = *ASTCtx.getTranslationUnitDecl();
    TypeDeclOrQualType ResType = Declarator.getOrCreateType(&I, ASTCtx, TUDecl);
    Res = new (ASTCtx) ParenExpr({}, {}, Res);
    Res = createCast(DeclCreator::getQualType(ResType), Res, ASTCtx);
  } break;
  default:
    break;
  }
  return Res;
}

Expr *StmtBuilder::getBoolLiteral(bool V) {
  QualType IntT = ASTCtx.IntTy;
  TypeDeclOrQualType BoolTy = Declarator.getOrCreateBoolType(ASTCtx);
  APInt Const = APInt(ASTCtx.getIntWidth(IntT), V ? 1 : 0, true);
  Expr *IntLiteral = IntegerLiteral::Create(ASTCtx, Const, IntT, {});
  return createCast(DeclCreator::getQualType(BoolTy), IntLiteral, ASTCtx);
}

Expr *StmtBuilder::getUIntLiteral(uint64_t U) {
  QualType UIntT = ASTCtx.UnsignedIntTy;
  APInt Const = APInt(ASTCtx.getIntWidth(UIntT), U);
  return IntegerLiteral::Create(ASTCtx, Const, UIntT, {});
}

Expr *StmtBuilder::getExprForValue(const Value *V) {
  revng_log(ASTBuildLog, "getExprForValue: " << dumpToString(V));

  if (auto *FunctionOrGlobal = dyn_cast<GlobalObject>(V)) {

    DeclaratorDecl *Decl = Declarator.globalDecls().at(FunctionOrGlobal);
    QualType Type = Decl->getType();
    DeclRefExpr *Res = new (ASTCtx)
      DeclRefExpr(ASTCtx, Decl, false, Type, VK_LValue, {});
    return Res;

  } else if (isa<llvm::ConstantAggregateZero>(V)
             or isa<llvm::ConstantStruct>(V)) {

    VarDecl *VDecl = VarDecls.at(V);
    QualType Type = VDecl->getType();
    DeclRefExpr *Res = new (ASTCtx)
      DeclRefExpr(ASTCtx, VDecl, false, Type, VK_LValue, {});
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

      const Value *Addr = nullptr;
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
                                          VK_RValue,
                                          FPOptions());

      clang::DeclContext &TUDecl = *ASTCtx.getTranslationUnitDecl();

      TypeDeclOrQualType PointeeType;

      if (Load) {
        PointeeType = Declarator.getOrCreateType(Load, ASTCtx, TUDecl);
      } else {
        const Value *Stored = Store->getValueOperand();
        PointeeType = Declarator.getOrCreateType(Stored, ASTCtx, TUDecl);
      }

      QualType PointeeQualType = DeclCreator::getQualType(PointeeType);

      QualAddrType = AddrExpr->getType();
      const clang::Type *AddrTy = QualAddrType.getTypePtr();
      if (not AddrTy->isPointerType()) {
        revng_assert(AddrTy->isBuiltinType());
        revng_assert(AddrTy->isIntegerType());

        QualType PtrTy = ASTCtx.getPointerType(PointeeQualType);
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

      return clang::UnaryOperator::Create(ASTCtx,
                                          AddrExpr,
                                          UnaryOperatorKind::UO_Deref,
                                          PointeeQualType,
                                          VK_LValue,
                                          OK_Ordinary,
                                          {},
                                          false,
                                          FPOptions());
    }
    if (auto *Cast = dyn_cast<CastInst>(I)) {
      Value *RHS = Cast->getOperand(0);
      Expr *Result = getParenthesizedExprForValue(RHS);
      LLVMType *RHSTy = Cast->getSrcTy();
      LLVMType *LHSTy = Cast->getDestTy();
      if (RHSTy != LHSTy) {
        revng_assert(RHSTy->isIntOrPtrTy() and LHSTy->isIntOrPtrTy());
        clang::DeclContext &TUDecl = *ASTCtx.getTranslationUnitDecl();
        TypeDeclOrQualType DestTy = Declarator.getOrCreateType(LHSTy,
                                                               nullptr,
                                                               ASTCtx,
                                                               TUDecl);
        QualType DestQualTy = DeclCreator::getQualType(DestTy);

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
          const clang::Type *PtrType = DestQualTy.getTypePtr();
          revng_assert(PtrType->isPointerType());
          uint64_t PtrSize = ASTCtx.getTypeSize(DestQualTy);
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

        TypeSourceInfo *TI = ASTCtx.CreateTypeSourceInfo(DestQualTy);
        Result = CStyleCastExpr::Create(ASTCtx,
                                        DestQualTy,
                                        VK_RValue,
                                        CK,
                                        Result,
                                        nullptr,
                                        FPOptions(),
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

    const llvm::Function *ArgFun = Arg->getParent();
    llvm::FunctionType *FType = ArgFun->getFunctionType();
    revng_assert(not FType->isVarArg());
    unsigned NumLLVMParams = FType->getNumParams();
    unsigned ArgNo = Arg->getArgNo();
    clang::FunctionDecl &FunDecl = Declarator.getFunctionDecl(ArgFun);
    unsigned DeclNumParams = FunDecl.getNumParams();
    revng_assert(NumLLVMParams == DeclNumParams);
    clang::ParmVarDecl *ParamVDecl = FunDecl.getParamDecl(ArgNo);
    QualType Type = ParamVDecl->getType();
    DeclRefExpr *Res = new (ASTCtx)
      DeclRefExpr(ASTCtx, ParamVDecl, false, Type, VK_LValue, {});
    return Res;

  } else {

    revng_abort();
  }
}

Expr *StmtBuilder::getLiteralFromConstant(const llvm::Constant *C) {
  if (auto *CD = dyn_cast<ConstantData>(C)) {
    if (auto *CInt = dyn_cast<ConstantInt>(CD)) {
      clang::DeclContext &TUDecl = *ASTCtx.getTranslationUnitDecl();
      TypeDeclOrQualType LiteralTy = Declarator.getOrCreateType(CInt->getType(),
                                                                nullptr,
                                                                ASTCtx,
                                                                TUDecl);
      QualType LiteralQualTy = DeclCreator::getQualType(LiteralTy);
      const clang::Type *UnderlyingTy = LiteralQualTy.getTypePtrOrNull();
      revng_assert(UnderlyingTy != nullptr);
      // Desugar stdint.h typedefs
      UnderlyingTy = UnderlyingTy->getUnqualifiedDesugaredType();
      const BuiltinType *BuiltinTy = cast<BuiltinType>(UnderlyingTy);
      switch (BuiltinTy->getKind()) {
      case BuiltinType::Bool: {
        QualType IntT = ASTCtx.IntTy;
        TypeDeclOrQualType
          BoolTy = Declarator.getOrCreateBoolType(ASTCtx, C->getType());
        uint64_t ConstValue = CInt->getValue().getZExtValue();
        APInt Const = APInt(ASTCtx.getIntWidth(IntT), ConstValue, true);
        Expr *IntLiteral = IntegerLiteral::Create(ASTCtx, Const, IntT, {});
        return createCast(DeclCreator::getQualType(BoolTy), IntLiteral, ASTCtx);
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
        APInt Const = APInt(ASTCtx.getIntWidth(LiteralQualTy), ConstValue);
        return IntegerLiteral::Create(ASTCtx, Const, LiteralQualTy, {});
      }
      case BuiltinType::Int:
      case BuiltinType::Long:
      case BuiltinType::LongLong: {
        uint64_t ConstValue = CInt->getValue().getZExtValue();
        APInt Const = APInt(ASTCtx.getIntWidth(LiteralQualTy),
                            ConstValue,
                            true);
        return IntegerLiteral::Create(ASTCtx, Const, LiteralQualTy, {});
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
      case BuiltinType::OMPArrayShaping:
      case BuiltinType::OMPIterator:
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
      case BuiltinType::BFloat16:
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
      case BuiltinType::SveBFloat16:
      case BuiltinType::SveFloat16:
      case BuiltinType::SveFloat32:
      case BuiltinType::SveFloat64:
      case BuiltinType::SveBFloat16x2:
      case BuiltinType::SveFloat16x2:
      case BuiltinType::SveFloat32x2:
      case BuiltinType::SveFloat64x2:
      case BuiltinType::SveBFloat16x3:
      case BuiltinType::SveFloat16x3:
      case BuiltinType::SveFloat32x3:
      case BuiltinType::SveFloat64x3:
      case BuiltinType::SveBFloat16x4:
      case BuiltinType::SveFloat16x4:
      case BuiltinType::SveFloat32x4:
      case BuiltinType::SveFloat64x4:
      case BuiltinType::SveInt8:
      case BuiltinType::SveInt16:
      case BuiltinType::SveInt32:
      case BuiltinType::SveInt64:
      case BuiltinType::SveInt8x2:
      case BuiltinType::SveInt16x2:
      case BuiltinType::SveInt32x2:
      case BuiltinType::SveInt64x2:
      case BuiltinType::SveInt8x3:
      case BuiltinType::SveInt16x3:
      case BuiltinType::SveInt32x3:
      case BuiltinType::SveInt64x3:
      case BuiltinType::SveInt8x4:
      case BuiltinType::SveInt16x4:
      case BuiltinType::SveInt32x4:
      case BuiltinType::SveInt64x4:
      case BuiltinType::SveUint8:
      case BuiltinType::SveUint16:
      case BuiltinType::SveUint32:
      case BuiltinType::SveUint64:
      case BuiltinType::SveUint8x2:
      case BuiltinType::SveUint16x2:
      case BuiltinType::SveUint32x2:
      case BuiltinType::SveUint64x2:
      case BuiltinType::SveUint8x3:
      case BuiltinType::SveUint16x3:
      case BuiltinType::SveUint32x3:
      case BuiltinType::SveUint64x3:
      case BuiltinType::SveUint8x4:
      case BuiltinType::SveUint16x4:
      case BuiltinType::SveUint32x4:
      case BuiltinType::SveUint64x4:
      case BuiltinType::VectorPair:
      case BuiltinType::VectorQuad:
      case BuiltinType::IncompleteMatrixIdx:
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
