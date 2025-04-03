#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/Support/GenericDomTree.h"

#include "revng/ABI/FunctionType/Layout.h"
#include "revng/ADT/RecursiveCoroutine.h"
#include "revng/ADT/ScopedExchange.h"
#include "revng/LocalVariables/LocalVariableHelpers.h"
#include "revng/Model/Binary.h"
#include "revng/Model/IRHelpers.h"
#include "revng/Pipeline/Location.h"
#include "revng/Pipes/Ranks.h"
#include "revng/RestructureCFG/ScopeGraphGraphTraits.h"
#include "revng/Support/FunctionTags.h"
#include "revng/mlir/Dialect/Clift/IR/CliftOps.h"
#include "revng/mlir/Dialect/Clift/Utils/ImportLLVM.h"
#include "revng/mlir/Dialect/Clift/Utils/ImportModel.h"

namespace clift = mlir::clift;
using namespace clift;

namespace {

[[nodiscard]] mlir::OpBuilder::InsertPoint
makeInsertionPointAfterStart(mlir::Block *Block) {
  return mlir::OpBuilder::InsertPoint(Block, Block->end());
}

[[nodiscard]] mlir::OpBuilder::InsertPoint
saveInsertionPointAfter(const mlir::OpBuilder &Builder) {
  mlir::Block *Block = Builder.getInsertionBlock();
  mlir::Block::iterator Point = Builder.getInsertionPoint();

  // End iterator specifies insertion at the start of Block. Non-end iterator
  // specifies insertion *after* the operation referred to by the iterator.
  Point = Point == Block->begin() ? Block->end() : std::prev(Point);

  return mlir::OpBuilder::InsertPoint(Block, Point);
}

void restoreInsertionPointAfter(mlir::OpBuilder &Builder,
                                mlir::OpBuilder::InsertPoint InsertPoint) {
  revng_assert(InsertPoint.isSet());

  mlir::Block *Block = InsertPoint.getBlock();
  mlir::Block::iterator Point = InsertPoint.getPoint();

  // Convert an end iterator back into begin and advance non-end iterators. This
  // is because the builder inserts operations *before* the specified iterator.
  Point = Point == Block->end() ? Block->begin() : std::next(Point);

  Builder.setInsertionPoint(Block, Point);
}

template<typename OpT, typename... ArgsT>
static mlir::OwningOpRef<OpT> createOperation(mlir::MLIRContext *Context,
                                              mlir::Location Location,
                                              ArgsT &&...Args) {
  OpT Op = mlir::OpBuilder(Context).create<OpT>(Location,
                                                std::forward<ArgsT>(Args)...);
  return mlir::OwningOpRef<OpT>(Op);
}

using ScopeGraphPostDomTree = llvm::PostDomTreeOnView<llvm::BasicBlock, Scope>;

class LLVMCodeImporter {
public:
  static void import(mlir::ModuleOp ModuleOp,
                     const model::Binary &Model,
                     const llvm::Module *Module) {
    LLVMCodeImporter(ModuleOp.getContext(), Model).importModule(ModuleOp,
                                                                Module);
  }

private:
  explicit LLVMCodeImporter(mlir::MLIRContext *Context,
                            const model::Binary &Model) :
    Context(Context),
    Model(Model),
    Builder(Context),
    PointerSize(getPointerSize(Model.Architecture())) {}

  /* Debug info */

  mlir::Location getLocation(const llvm::DISubprogram *Subprogram) {
    if (Subprogram == nullptr)
      return mlir::UnknownLoc::get(Context);

    auto Content = mlir::StringAttr::get(Context, Subprogram->getName());
    return mlir::NameLoc::get(Content, mlir::UnknownLoc::get(Context));
  }

  mlir::Location getLocation(const llvm::DebugLoc &DL) {
    if (not DL)
      return mlir::UnknownLoc::get(Context);

    const llvm::MDNode *Scope = DL->getScope();
    return getLocation(llvm::dyn_cast_or_null<llvm::DISubprogram>(Scope));
  }

  mlir::Location getLocation(const llvm::BasicBlock *BB) {
    return getLocation(&BB->front());
  }

  mlir::Location getLocation(const llvm::Instruction *I) {
    return getLocation(I->getDebugLoc());
  }

  /* Type utilities */

  static uint64_t getIntegerSize(unsigned IntegerWidth) {
    // Compute the smallest power-of-two number of bytes capable of representing
    // the type based on its bit width:
    return std::bit_ceil((IntegerWidth + 7) / 8);
  }

  clift::ValueType
  getPrimitiveType(uint64_t Size,
                   PrimitiveKind Kind = PrimitiveKind::GenericKind) {
    return PrimitiveType::get(Context, Kind, Size);
  }

  clift::ValueType makePointerType(clift::ValueType ElementType) {
    return PointerType::get(ElementType, PointerSize);
  }

  clift::ValueType getVoidType() {
    if (not VoidTypeCache)
      VoidTypeCache = getPrimitiveType(0, PrimitiveKind::VoidKind);
    return VoidTypeCache;
  }

  clift::ValueType getVoidPointerType() {
    if (not PointerTypeCache)
      PointerTypeCache = makePointerType(getVoidType());
    return PointerTypeCache;
  }

  clift::ValueType getIntptrType() {
    if (not IntptrTypeCache)
      IntptrTypeCache = getPrimitiveType(PointerSize);
    return IntptrTypeCache;
  }

  /* Model type import */

  clift::ValueType importModelType(const model::Type &Type) {
    auto EmitError = [&]() -> mlir::InFlightDiagnostic {
      return Context->getDiagEngine().emit(mlir::UnknownLoc::get(Context),
                                           mlir::DiagnosticSeverity::Error);
    };
    return clift::importModelType(EmitError, *Context, Type, Model);
  }

  clift::ValueType importModelType(const model::TypeDefinition &Type) {
    auto EmitError = [&]() -> mlir::InFlightDiagnostic {
      return Context->getDiagEngine().emit(mlir::UnknownLoc::get(Context),
                                           mlir::DiagnosticSeverity::Error);
    };
    return clift::importModelType(EmitError, *Context, Type, Model);
  }

  template<typename TypeT, typename ModelTypeT>
  TypeT importModelType(const ModelTypeT &Type) {
    return mlir::cast<TypeT>(importModelType(Type));
  }

  /* LLVM type import */

  clift::ValueType
  importLLVMIntegerType(const llvm::IntegerType *Type,
                        PrimitiveKind Kind = PrimitiveKind::GenericKind) {
    return getPrimitiveType(getIntegerSize(Type->getBitWidth()), Kind);
  }

  clift::ValueType importLLVMPointerType(const llvm::PointerType *Type) {
    revng_assert(Type->getAddressSpace() == 0);
    return getVoidPointerType();
  }

  clift::ValueType importLLVMArrayType(const llvm::ArrayType *Type) {
    auto ElementType = importLLVMType(Type->getElementType());
    return ArrayType::get(ElementType, Type->getNumElements());
  }

  clift::ValueType importLLVMType(const llvm::Type *Type) {
    if (Type->isVoidTy())
      return getVoidType();

    if (auto *T = llvm::dyn_cast<llvm::IntegerType>(Type))
      return importLLVMIntegerType(T);

    if (auto *T = llvm::dyn_cast<llvm::PointerType>(Type))
      return importLLVMPointerType(T);

    if (auto *T = llvm::dyn_cast<llvm::ArrayType>(Type))
      return importLLVMArrayType(T);

    revng_abort("Unsupported LLVM type");
  }

  /* LLVM expression import */

  static bool
  isByAddressParameter(const abi::FunctionType::Layout::Argument &Parameter) {
    switch (Parameter.Kind) {
    case abi::FunctionType::ArgumentKind::PointerToCopy:
    case abi::FunctionType::ArgumentKind::ReferenceToAggregate:
      return true;
    default:
      return false;
    }
  }

  uint64_t getConstantInt(const llvm::Value *Value) {
    return llvm::cast<llvm::ConstantInt>(Value)->getZExtValue();
  }

  clift::ValueType importHelperReturnType(const llvm::Type *Type,
                                          llvm::StringRef HelperName) {
    auto StructType = llvm::dyn_cast<llvm::StructType>(Type);

    if (not StructType)
      return importLLVMType(Type);

    revng_assert(StructType->isLiteral());

    llvm::SmallVector<FieldAttr> Fields;
    Fields.reserve(StructType->getNumElements());

    uint64_t Offset = 0;
    for (const llvm::Type *T : StructType->elements()) {
      clift::ValueType FieldType = importLLVMType(T);
      Fields.push_back(FieldAttr::get(Context, Offset, FieldType, ""));
      Offset += FieldType.getByteSize();
    }

    auto Handle = pipeline::locationString(revng::ranks::HelperStructType,
                                           HelperName.str());

    return StructType::get(Context, Handle, "", Offset, Fields);
  }

  clift::FunctionType importHelperType(const llvm::FunctionType *Type,
                                       llvm::StringRef HelperName) {
    revng_assert(not Type->isVarArg());

    clift::ValueType ReturnType = importHelperReturnType(Type->getReturnType(),
                                                         HelperName);

    llvm::SmallVector<clift::ValueType> ParameterTypes;
    ParameterTypes.reserve(Type->getNumParams());

    for (const llvm::Type *T : Type->params())
      ParameterTypes.push_back(importLLVMType(T));

    auto Handle = pipeline::locationString(revng::ranks::HelperFunction,
                                           HelperName.str());

    return FunctionType::get(Context, Handle, "", ReturnType, ParameterTypes);
  }

  clift::FunctionType importHelperType(const llvm::Function *F) {
    return importHelperType(F->getFunctionType(), F->getName());
  }

  clift::FunctionOp emitHelperDeclaration(mlir::Location Loc,
                                          llvm::StringRef HelperName,
                                          clift::FunctionType FunctionType) {
    auto [Iterator, Inserted] = HelperMapping.try_emplace(HelperName);

    if (Inserted) {
      mlir::OpBuilder::InsertionGuard Guard(Builder);
      Builder.setInsertionPointToEnd(CurrentModule.getBody());

      auto Op = Builder.create<FunctionOp>(Loc, HelperName, FunctionType);

      Op.setHandle(pipeline::locationString(revng::ranks::HelperFunction,
                                            HelperName.str()));

      Iterator->second = Op;
    }

    return Iterator->second;
  }

  clift::FunctionOp emitHelperDeclaration(const llvm::Function *F) {
    return emitHelperDeclaration(getLocation(F->getSubprogram()),
                                 F->getName(),
                                 importHelperType(F));
  }

  clift::FunctionOp emitModelFunctionDeclaration(const llvm::Function *F,
                                                    const model::Function *MF) {
    auto FunctionType = importModelType<clift::FunctionType>(*MF->Prototype());

    auto Op = Builder.create<FunctionOp>(getLocation(F->getSubprogram()),
                                         F->getName(),
                                         FunctionType);

    Op.setHandle(pipeline::locationString(revng::ranks::Function, MF->Entry()));

    return Op;
  }

  clift::FunctionOp emitFunctionDeclaration(const llvm::Function *F) {
    if (const model::Function *MF = llvmToModelFunction(Model, *F))
      return emitModelFunctionDeclaration(F, MF);
    else
      return emitHelperDeclaration(F);
  }

  auto getOrEmitSymbol(const llvm::GlobalObject *G, auto Emit) {
    using ReturnType = std::invoke_result_t<decltype(Emit)>;
    auto [Iterator, Inserted] = SymbolMapping.try_emplace(G);
    if (Inserted)
      Iterator->second = Emit();
    return mlir::cast<ReturnType>(Iterator->second);
  }

  clift::ValueType emitGlobalObject(const llvm::GlobalObject *G) {
    auto Op = getOrEmitSymbol(G, [&]() -> clift::GlobalOpInterface {
      mlir::OpBuilder::InsertionGuard Guard(Builder);
      Builder.setInsertionPointToEnd(CurrentModule.getBody());

      if (auto *F = llvm::dyn_cast<llvm::Function>(G))
        return emitFunctionDeclaration(F);

      llvm::errs() << "G: " << *G << "\n";
      revng_abort("Unsupported global object kind");
    });

    return Op.getType();
  }

  template<typename OpT, typename... ArgsT>
  mlir::Value emitExpr(mlir::Location Loc, ArgsT &&...Args) {
    return Builder.create<OpT>(Loc, std::forward<ArgsT>(Args)...);
  }

  mlir::Value emitCast(mlir::Location Loc,
                       mlir::Value Value,
                       clift::ValueType TargetType,
                       CastKind Kind = CastKind::Bitcast) {
    if (Value.getType() != TargetType)
      Value = Builder.create<CastOp>(Loc, TargetType, Value, Kind);

    return Value;
  }

  mlir::Value emitImplicitCast(mlir::Location Loc,
                               mlir::Value Value,
                               clift::ValueType TargetType) {
    auto SourceType = mlir::cast<clift::ValueType>(Value.getType());

    if (SourceType == TargetType)
      return Value;

    auto UnderlyingSourceT = dealias(SourceType, true);
    auto UnderlyingTargetT = dealias(TargetType, true);

    if (UnderlyingSourceT.getByteSize() != UnderlyingTargetT.getByteSize())
      return Value;

    if (mlir::isa<ArrayType>(UnderlyingSourceT))
      return Value;
    if (mlir::isa<ArrayType>(UnderlyingTargetT))
      return Value;

    return emitCast(Loc, Value, TargetType);
  }

  mlir::Value emitIntegerOp(mlir::Location Loc,
                            PrimitiveKind Kind,
                            auto ApplyOperation,
                            std::same_as<mlir::Value> auto... Operands) {
    auto ConvertToKind = [&](mlir::Value &Value, PrimitiveKind Kind) {
      auto UnderlyingType = getUnderlyingIntegerType(Value.getType());
      if (not UnderlyingType)
        llvm::errs() << Value.getType() << "\n";
      revng_assert(UnderlyingType);

      uint64_t Size = UnderlyingType.getSize();
      Value = emitCast(Loc, Value, getPrimitiveType(Size, Kind));
    };

    (ConvertToKind(Operands, Kind), ...);
    mlir::Value Result = ApplyOperation(Operands...);

    if (Kind != PrimitiveKind::GenericKind)
      ConvertToKind(Result, PrimitiveKind::GenericKind);

    return Result;
  }

  mlir::Value emitIntegerCast(mlir::Location Loc,
                              mlir::Value Operand,
                              uint64_t Size,
                              PrimitiveKind Kind) {
    uint64_t SrcSize = getUnderlyingIntegerType(Operand.getType()).getSize();

    CastKind Cast = CastKind::Bitcast;
    if (Size > SrcSize)
      Cast = CastKind::Extend;
    if (Size < SrcSize)
      Cast = CastKind::Truncate;

    auto EmitCast = [&](mlir::Value Operand) {
      return emitCast(Loc, Operand, getPrimitiveType(Size, Kind), Cast);
    };

    return emitIntegerOp(Loc, Kind, EmitCast, Operand);
  }

  FunctionOp getHelperFunction(llvm::StringRef HelperName,
                               clift::FunctionType FunctionType) {
    return emitHelperDeclaration(mlir::UnknownLoc::get(Context),
                                 HelperName,
                                 FunctionType);
  }

  FunctionOp getHelperFunction(const llvm::Function *F) {
    return getOrEmitSymbol(F, [&]() -> FunctionOp {
      return emitHelperDeclaration(F);
    });
  }

  mlir::Value useGlobal(mlir::Location Loc, clift::GlobalOpInterface Global) {
    return Builder.create<UseOp>(Loc, Global.getType(), Global.getName());
  }

  mlir::Value emitHelperCall(mlir::Location Loc,
                             FunctionOp Function,
                             llvm::ArrayRef<mlir::Value> Arguments) {
    auto FunctionType = Function.getCliftFunctionType();

    llvm::SmallVector<mlir::Value> CastArgs;
    CastArgs.reserve(Arguments.size());

    for (auto [A, T] : llvm::zip(Arguments, FunctionType.getArgumentTypes()))
      CastArgs.push_back(emitImplicitCast(Loc, A, T));

    return Builder.create<CallOp>(Loc,
                                  FunctionType.getReturnType(),
                                  useGlobal(Loc, Function),
                                  CastArgs);
  }

  mlir::Value emitHelperCall(mlir::Location Loc,
                             const llvm::Function *F,
                             llvm::ArrayRef<mlir::Value> Arguments) {
    return emitHelperCall(Loc, getHelperFunction(F), Arguments);
  }

  mlir::Value emitHelperCall(mlir::Location Loc,
                             llvm::StringRef HelperName,
                             mlir::Type ReturnType,
                             llvm::ArrayRef<mlir::Type> ParameterTypes,
                             llvm::ArrayRef<mlir::Value> Arguments) {
    auto FunctionType = clift::FunctionType::get(Context,
                                                 "",
                                                 "",
                                                 ReturnType,
                                                 ParameterTypes);

    return emitHelperCall(Loc,
                          getHelperFunction(HelperName, FunctionType),
                          Arguments);
  }

  RecursiveCoroutine<mlir::Value>
  emitOpaqueExtractValue(const llvm::CallInst *Call) {
    mlir::Location Loc = getLocation(Call);

    mlir::Value Aggregate = rc_recur emitExpression(Call->getArgOperand(0),
                                                    Loc);

    uint64_t Index = getConstantInt(Call->getArgOperand(1));
    auto Struct = mlir::cast<StructType>(Aggregate.getType());
    revng_assert(Index < Struct.getFields().size());
    mlir::Type ResultType = Struct.getFields()[Index].getType();

    rc_return Builder.create<AccessOp>(Loc,
                                       ResultType,
                                       Aggregate,
                                       /*indirect=*/false,
                                       Index);
  }

  RecursiveCoroutine<mlir::Value> emitHelperCall(const llvm::CallInst *Call) {
    const llvm::Function *Function = Call->getCalledFunction();
    auto Tags = FunctionTags::TagsSet::from(Function);

    if (Tags.contains(FunctionTags::OpaqueExtractValue))
      rc_return rc_recur emitOpaqueExtractValue(Call);

    mlir::Location Loc = getLocation(Call);

    llvm::SmallVector<mlir::Value> Arguments;
    Arguments.reserve(Call->arg_size());

    for (const llvm::Value *Argument : Call->args())
      Arguments.push_back(rc_recur emitExpression(Argument, Loc));

    rc_return emitHelperCall(Loc, getHelperFunction(Function), Arguments);
  }

  struct StringLiteral {
    clift::ArrayType Type;
    std::string Data;
  };

  bool detectStringLiteralImpl(const llvm::GlobalVariable *V,
                               StringLiteral& String) {
    if (not V->isConstant())
      return false;

    const auto *Type = llvm::dyn_cast<llvm::ArrayType>(V->getValueType());
    if (Type == nullptr)
      return false;

    const auto *ET = llvm::dyn_cast<llvm::IntegerType>(Type->getElementType());
    if (ET == nullptr or ET->getBitWidth() != 8)
      return false;

    const llvm::Constant *Init = V->getInitializer();
    if (Init == nullptr)
      return false;
    revng_assert(Init->getType() == Type);

    const auto *Array = llvm::dyn_cast<llvm::ConstantDataArray>(Init);
    if (Array == nullptr and not llvm::isa<llvm::ConstantAggregateZero>(Init))
      return false;

    revng_assert(Type->getNumElements() != 0);
    unsigned Length = Type->getNumElements() - 1;

    const auto GetChar = [&](unsigned Index) -> uint8_t {
      uint64_t Value = Array ?
        getConstantInt(Array->getAggregateElement(Index)) :
        0;
      revng_assert(Value < 0x100);
      return Value;
    };

    if (GetChar(Length) != 0)
      return false;

    auto CharType = clift::PrimitiveType::get(Context,
                                              PrimitiveKind::NumberKind,
                                              1,
                                              /*IsConst=*/true);

    String.Type = clift::ArrayType::get(CharType, Type->getNumElements());

    String.Data.reserve(Length);
    for (unsigned I = 0; I < Length; ++I)
      String.Data.push_back(GetChar(I));

    return true;
  }

  const StringLiteral *detectStringLiteral(const llvm::GlobalObject *O) {
    if (const auto *V = llvm::dyn_cast<llvm::GlobalVariable>(O)) {
      auto [Iterator, Inserted] = StringMapping.try_emplace(V);
      if (not Inserted or detectStringLiteralImpl(V, Iterator->second))
        return &Iterator->second;

      StringMapping.erase(Iterator);
    }
    return nullptr;
  }

  RecursiveCoroutine<mlir::Value>
  emitExpression(const llvm::Value *V, mlir::Location SurroundingLocation) {
    if (auto G = llvm::dyn_cast<llvm::GlobalObject>(V)) {
      if (const StringLiteral *String = detectStringLiteral(G)) {
        auto Op = Builder.create<StringOp>(SurroundingLocation,
                                           String->Type,
                                           std::move(String->Data));

        auto Type = makePointerType(String->Type.getElementType());

        rc_return emitCast(SurroundingLocation, Op, Type, CastKind::Decay);
      }

      rc_return Builder.create<UseOp>(SurroundingLocation,
                                      emitGlobalObject(G),
                                      G->getName());
    }

    if (auto It = ValueMapping.find(V); It != ValueMapping.end()) {
      mlir::Value Value = It->second.Value;

      if (It->second.Addressof) {
        Value = Builder.create<AddressofOp>(SurroundingLocation,
                                            makePointerType(Value.getType()),
                                            Value);
      }

      rc_return Value;
    }

    if (auto A = llvm::dyn_cast<llvm::Argument>(V)) {
      auto It = ArgumentMapping.find(A);
      revng_assert(It != ArgumentMapping.end());

      mlir::Value Value = It->second.Argument;

      if (It->second.IsByAddress)
        Value = Builder.create<AddressofOp>(SurroundingLocation,
                                            makePointerType(Value.getType()),
                                            Value);

      rc_return emitImplicitCast(SurroundingLocation,
                                Value,
                                It->second.CastType);
    }

    if (auto U = llvm::dyn_cast<llvm::UndefValue>(V)) {
      mlir::Type Type = importLLVMType(U->getType());
      rc_return Builder.create<UndefOp>(SurroundingLocation, Type);
    }

    if (auto C = llvm::dyn_cast<llvm::ConstantInt>(V)) {
      const llvm::IntegerType *T = llvm::cast<llvm::IntegerType>(C->getType());
      rc_return Builder.create<ImmediateOp>(SurroundingLocation,
                                            importLLVMIntegerType(T),
                                            C->getZExtValue());
    }

    if (auto N = llvm::dyn_cast<llvm::ConstantPointerNull>(V)) {
      auto Op = Builder.create<ImmediateOp>(SurroundingLocation,
                                            getIntptrType(),
                                            /*Value=*/0);

      rc_return emitCast(SurroundingLocation, Op, getVoidPointerType());
    }

    if (auto E = llvm::dyn_cast<llvm::ConstantExpr>(V)) {
      // TODO: This could be made more efficient by implementing direct
      //       ConstantExpr import.
      llvm::Instruction *I = E->getAsInstruction();
      mlir::Value Value = emitExpression(I, SurroundingLocation);
      I->deleteValue();
      rc_return Value;
    }

    if (auto I = llvm::dyn_cast<llvm::AllocaInst>(V)) {
      mlir::Location Loc = SurroundingLocation;

      if (auto It = AllocaMapping.find(I); It != AllocaMapping.end()) {
        auto Type = makePointerType(It->second.getType());
        auto Value = Builder.create<AddressofOp>(Loc, Type, It->second);
        rc_return emitImplicitCast(Loc, Value, importLLVMType(I->getType()));
      }

      revng_assert(not llvm::isa<llvm::Constant>(I->getArraySize()));
      revng_abort("Non-constant alloca is not supported.");
    }

    if (auto I = llvm::dyn_cast<llvm::LoadInst>(V)) {
      mlir::Location Loc = getLocation(I);

      mlir::Value Pointer = rc_recur emitExpression(I->getPointerOperand(),
                                                    Loc);

      clift::ValueType ValueType = importLLVMType(V->getType());
      clift::ValueType PointerType = makePointerType(ValueType);

      auto Op1 = Builder.create<CastOp>(Loc,
                                        PointerType,
                                        Pointer,
                                        CastKind::Bitcast);

      rc_return Builder.create<IndirectionOp>(Loc, ValueType, Op1);
    }

    if (auto I = llvm::dyn_cast<llvm::StoreInst>(V)) {
      mlir::Location Loc = getLocation(I);

      mlir::Value Pointer = rc_recur emitExpression(I->getPointerOperand(),
                                                    Loc);
      mlir::Value Value = rc_recur emitExpression(I->getValueOperand(), Loc);

      auto Op1 = Builder.create<CastOp>(Loc,
                                        makePointerType(Value.getType()),
                                        Pointer,
                                        CastKind::Bitcast);

      auto Op2 = Builder.create<IndirectionOp>(Loc, Value.getType(), Op1);

      rc_return Builder.create<AssignOp>(Loc, Value.getType(), Op2, Value);
    }

    if (auto I = llvm::dyn_cast<llvm::BinaryOperator>(V)) {
      using Operators = llvm::BinaryOperator::BinaryOps;

      mlir::Location Loc = getLocation(I);

      mlir::Value Lhs = rc_recur emitExpression(I->getOperand(0), Loc);
      mlir::Value Rhs = rc_recur emitExpression(I->getOperand(1), Loc);

#if 0
      auto LhsPointerType = mlir::dyn_cast<PointerType>(Lhs.getType());
      auto RhsPointerType = mlir::dyn_cast<PointerType>(Rhs.getType());

      if (LhsPointerType or RhsPointerType) {
        switch (I->getOpcode()) {
        case Operators::Add: {
          revng_assert(not LhsPointerType or not RhsPointerType);
          auto Type = LhsPointerType ? LhsPointerType : RhsPointerType;
          rc_return emitExpr<PtrAddOp>(Loc, Type, Lhs, Rhs);
        }

        case Operators::Sub:
          if (LhsPointerType and RhsPointerType) {
            revng_assert(LhsPointerType == RhsPointerType);
            auto Type = getPrimitiveType(LhsPointerType.getPointerSize(),
                                         PrimitiveKind::SignedKind);
            rc_return emitExpr<PtrDiffOp>(Loc, Type, Lhs, Rhs);
          } else {
            auto Type = LhsPointerType ? LhsPointerType : RhsPointerType;
            rc_return emitExpr<PtrSubOp>(Loc, Type, Lhs, Rhs);
          }

        default:
          revng_abort("Unsupported pointer arithmetic operation.");
        }
      }
#endif

      PrimitiveKind Kind = PrimitiveKind::GenericKind;
      switch (I->getOpcode()) {
      case Operators::SDiv:
      case Operators::SRem:
      case Operators::AShr:
        Kind = PrimitiveKind::SignedKind;
        break;
      case Operators::UDiv:
      case Operators::URem:
      case Operators::LShr:
        Kind = PrimitiveKind::UnsignedKind;
        break;
      default:
        break;
      }

      auto *IntegerType = llvm::cast<llvm::IntegerType>(V->getType());
      auto Type = importLLVMIntegerType(IntegerType, Kind);

      auto EmitOp = [&](mlir::Value Lhs, mlir::Value Rhs) {
        switch (I->getOpcode()) {
        case Operators::Add:
          return emitExpr<AddOp>(Loc, Type, Lhs, Rhs);
        case Operators::Sub:
          return emitExpr<SubOp>(Loc, Type, Lhs, Rhs);
        case Operators::Mul:
          return emitExpr<MulOp>(Loc, Type, Lhs, Rhs);
        case Operators::SDiv:
        case Operators::UDiv:
          return emitExpr<DivOp>(Loc, Type, Lhs, Rhs);
        case Operators::SRem:
        case Operators::URem:
          return emitExpr<RemOp>(Loc, Type, Lhs, Rhs);
        case Operators::Shl:
          return emitExpr<ShiftLeftOp>(Loc, Type, Lhs, Rhs);
        case Operators::LShr:
        case Operators::AShr:
          return emitExpr<ShiftRightOp>(Loc, Type, Lhs, Rhs);
        case Operators::And:
          return emitExpr<BitwiseAndOp>(Loc, Type, Lhs, Rhs);
        case Operators::Or:
          return emitExpr<BitwiseOrOp>(Loc, Type, Lhs, Rhs);
        case Operators::Xor:
          return emitExpr<BitwiseXorOp>(Loc, Type, Lhs, Rhs);
        default:
          revng_abort("Unsupported LLVM binary operator.");
        }
      };

      rc_return emitIntegerOp(Loc, Kind, EmitOp, Lhs, Rhs);
    }

    if (auto I = llvm::dyn_cast<llvm::ICmpInst>(V)) {
      using enum llvm::ICmpInst::Predicate;

      PrimitiveKind Kind = PrimitiveKind::GenericKind;
      switch (I->getPredicate()) {
      case ICMP_SGT:
      case ICMP_SGE:
      case ICMP_SLT:
      case ICMP_SLE:
        Kind = PrimitiveKind::SignedKind;
        break;
      case ICMP_UGT:
      case ICMP_UGE:
      case ICMP_ULT:
      case ICMP_ULE:
        Kind = PrimitiveKind::UnsignedKind;
        break;
      default:
        break;
      }

      mlir::Location Loc = getLocation(I);
      mlir::Value Lhs = rc_recur emitExpression(I->getOperand(0), Loc);
      mlir::Value Rhs = rc_recur emitExpression(I->getOperand(1), Loc);

      auto *IntegerType = llvm::cast<llvm::IntegerType>(V->getType());
      auto Type = importLLVMIntegerType(IntegerType, PrimitiveKind::SignedKind);

      auto EmitOp = [&](mlir::Value Lhs, mlir::Value Rhs) {
        switch (I->getPredicate()) {
        case ICMP_EQ:
          return emitExpr<CmpEqOp>(Loc, Type, Lhs, Rhs);
        case ICMP_NE:
          return emitExpr<CmpNeOp>(Loc, Type, Lhs, Rhs);
        case ICMP_SGT:
        case ICMP_UGT:
          return emitExpr<CmpGtOp>(Loc, Type, Lhs, Rhs);
        case ICMP_SGE:
        case ICMP_UGE:
          return emitExpr<CmpGeOp>(Loc, Type, Lhs, Rhs);
        case ICMP_SLT:
        case ICMP_ULT:
          return emitExpr<CmpLtOp>(Loc, Type, Lhs, Rhs);
        case ICMP_SLE:
        case ICMP_ULE:
          return emitExpr<CmpLeOp>(Loc, Type, Lhs, Rhs);
        default:
          revng_abort("Unsupported LLVM comparison predicate.");
        }
      };

      rc_return emitIntegerOp(Loc, Kind, EmitOp, Lhs, Rhs);
    }

    if (auto I = llvm::dyn_cast<llvm::CastInst>(V)) {
      mlir::Location Loc = getLocation(I);
      mlir::Value Operand = rc_recur emitExpression(I->getOperand(0), Loc);

      auto getIntegerSize = [](const llvm::Type *Type) {
        const auto *IntegerType = llvm::cast<llvm::IntegerType>(Type);
        return LLVMCodeImporter::getIntegerSize(IntegerType->getBitWidth());
      };

      auto emitIntegerCast = [&](PrimitiveKind Kind) {
        return this->emitIntegerCast(Loc,
                                     Operand,
                                     getIntegerSize(V->getType()),
                                     Kind);
      };

      //llvm::errs() << *I << "\n";
      //llvm::errs() << "  " << Operand.getType() << "\n";

      switch (I->getOpcode()) {
        using Operators = llvm::CastInst::CastOps;
      case Operators::Trunc:
        rc_return emitIntegerCast(PrimitiveKind::GenericKind);
      case Operators::SExt:
        rc_return emitIntegerCast(PrimitiveKind::SignedKind);
      case Operators::ZExt:
        rc_return emitIntegerCast(PrimitiveKind::UnsignedKind);
      case Operators::PtrToInt:
        Operand = emitCast(Loc, Operand, getIntptrType());
        if (uint64_t S = getIntegerSize(I->getDestTy()); S > PointerSize) {
          Operand = emitCast(Loc,
                             Operand,
                             getPrimitiveType(S),
                             CastKind::Extend);
        }
        rc_return Operand;
      case Operators::IntToPtr:
        if (getIntegerSize(I->getSrcTy()) > PointerSize) {
          Operand = emitCast(Loc,
                             Operand,
                             getPrimitiveType(PointerSize),
                             CastKind::Truncate);
        }

        rc_return emitCast(Loc, Operand, getVoidPointerType());
      default:
        revng_abort("Unsupported LLVM cast operation.");
      }
    }

    if (auto I = llvm::dyn_cast<llvm::CallInst>(V)) {
      mlir::Location Loc = getLocation(I);

      if (not I->hasMetadata(PrototypeMDName))
        rc_return emitHelperCall(I);

      const auto *ModelCallType = getCallSitePrototype(Model, I);
      auto Layout = abi::FunctionType::Layout::make(*ModelCallType);

      auto CallType = importModelType<clift::FunctionType>(*ModelCallType);

      mlir::Value Function = rc_recur emitExpression(I->getCalledOperand(),
                                                     Loc);

      clift::FunctionType
        FunctionType = getFunctionOrFunctionPointerFunctionType(Function
                                                                  .getType());

      if (CallType != FunctionType) {
        if (mlir::isa<clift::FunctionType>(Function.getType())) {
          Function = emitCast(Loc,
                              Function,
                              makePointerType(FunctionType),
                              CastKind::Decay);
        }

        Function = emitCast(Loc, Function, makePointerType(CallType));
      }

      llvm::ArrayRef LayoutArguments = getLayoutArguments(Layout);
      revng_assert(I->arg_size() == LayoutArguments.size());
      revng_assert(I->arg_size() == CallType.getArgumentTypes().size());

      llvm::SmallVector<mlir::Value> Arguments;
      for (auto [A, T, L] : llvm::zip(I->args(),
                                      CallType.getArgumentTypes(),
                                      LayoutArguments)) {
        mlir::Value Argument = rc_recur emitExpression(A, Loc);

        if (isByAddressParameter(L)) {
          Argument = emitImplicitCast(Loc, Argument, makePointerType(T));
          Argument = Builder.create<IndirectionOp>(Loc, T, Argument);
        } else {
          Argument = emitImplicitCast(Loc, Argument, T);
        }

        Arguments.push_back(Argument);
      }

      rc_return Builder.create<CallOp>(Loc,
                                       CallType.getReturnType(),
                                       Function,
                                       Arguments);
    }

    if (auto I = llvm::dyn_cast<llvm::SelectInst>(V)) {
      mlir::Location Loc = getLocation(I);

      mlir::Value Condition = rc_recur emitExpression(I->getCondition(), Loc);
      mlir::Value True = rc_recur emitExpression(I->getTrueValue(), Loc);
      mlir::Value False = rc_recur emitExpression(I->getFalseValue(), Loc);

      auto TrueType = mlir::cast<clift::ValueType>(True.getType());
      auto FalseType = mlir::cast<clift::ValueType>(False.getType());

      auto ResultType = TrueType.removeConst();
      if (ResultType != FalseType.removeConst()) {
        ResultType = importLLVMType(I->getType());
        True = emitImplicitCast(Loc, True, ResultType);
        False = emitImplicitCast(Loc, False, ResultType);
      }

      rc_return Builder.create<TernaryOp>(Loc,
                                          ResultType,
                                          Condition,
                                          True,
                                          False);
    }

    if (auto I = llvm::dyn_cast<llvm::GetElementPtrInst>(V)) {
      auto Alloca = llvm::cast<llvm::AllocaInst>(I->getPointerOperand());

      auto It = AllocaMapping.find(Alloca);
      revng_assert(It != AllocaMapping.end());

      auto AT = mlir::cast<clift::ArrayType>(It->second.getType());
      auto PT = makePointerType(AT.getElementType());

      revng_assert(I->getNumIndices() == 2);
      auto IndexIterator = I->idx_begin();

      revng_assert(getConstantInt(IndexIterator->get()) == 0);
      uint64_t Index1 = getConstantInt((++IndexIterator)->get());

      mlir::Location Loc = getLocation(I);
      auto Operand = emitCast(Loc, It->second, PT, CastKind::Decay);

      mlir::Value Immediate = emitExpr<ImmediateOp>(Loc,
                                                    getIntptrType(),
                                                    Index1);

      rc_return emitExpr<PtrAddOp>(Loc, PT, Operand, Immediate);
    }

    if (auto I = llvm::dyn_cast<llvm::FreezeInst>(V))
      rc_return emitExpression(I->getOperand(0), getLocation(I));

    revng_abort("Unsupported LLVM instruction.");
  }

  mlir::Type emitExpressionTreeImpl(mlir::Block &B, auto EmitExpression) {
    revng_assert(B.empty());

    mlir::OpBuilder::InsertionGuard Guard(Builder);
    Builder.setInsertionPointToEnd(&B);

    mlir::Value Value = EmitExpression();
    Builder.create<YieldOp>(Value.getLoc(), Value);
    return Value.getType();
  }

  mlir::Type emitExpressionTreeImpl(mlir::Region &R, auto EmitExpression) {
    revng_assert(R.empty());
    return emitExpressionTreeImpl(R.emplaceBlock(), EmitExpression);
  }

  mlir::Type emitExpressionTree(mlir::Block &B,
                                const llvm::Value *V,
                                mlir::Location SurroundingLocation) {
    return emitExpressionTreeImpl(B, [&]() {
      return emitExpression(V, SurroundingLocation);
    });
  }

  mlir::Type emitExpressionTree(mlir::Region &R,
                                const llvm::Value *V,
                                mlir::Location SurroundingLocation) {
    return emitExpressionTreeImpl(R, [&]() {
      return emitExpression(V, SurroundingLocation);
    });
  }

  [[nodiscard]] std::pair<mlir::Block *, mlir::Type>
  emitExpressionTree(const llvm::Value *V, mlir::Location SurroundingLocation) {
    mlir::Block *B = new mlir::Block();
    mlir::Type Type = emitExpressionTree(*B, V, SurroundingLocation);
    return { B, Type };
  }

  /* LLVM control flow import */

  template<typename OpT, typename ...ArgsT>
  OpT emitLocalDeclaration(mlir::Location Loc, ArgsT &&...Args) {
    mlir::OpBuilder::InsertionGuard Guard(Builder);
    restoreInsertionPointAfter(Builder, LocalDeclarationInsertPoint);
    OpT Op = Builder.create<OpT>(Loc, std::forward<ArgsT>(Args)...);
    LocalDeclarationInsertPoint = saveInsertionPointAfter(Builder);
    return Op;
  }

  void emitAssignLabel(mlir::Value Label, mlir::Location Loc) {
    Builder.create<AssignLabelOp>(Loc, Label);
  }

  static bool isScopeGraphEdge(const llvm::BasicBlock *Pred,
                               const llvm::BasicBlock *Succ) {
    for (auto *BB : llvm::children<Scope<const llvm::BasicBlock *>>(Pred)) {
      if (BB == Succ)
        return true;
    }
    return false;
  }

  static bool isUsedOutsideOfBlock(const llvm::Value *V,
                                   const llvm::BasicBlock *BB) {
    for (const llvm::User *U : V->users()) {
      if (auto *I = llvm::dyn_cast<llvm::Instruction>(U)) {
        if (I->getParent() != BB)
          return false;
      }
    }
    return true;
  }

  bool hasBeenEmitted(const llvm::BasicBlock *BB) {
    auto It = BlockMapping.find(BB);
    return It != BlockMapping.end() and It->second.InsertPoint.isSet();
  }

  void emitGoto(mlir::Location Loc, const llvm::BasicBlock *BB) {
    auto [Iterator, Inserted] = BlockMapping.try_emplace(BB);

    if (not Iterator->second.Label) {
      Iterator->second.Label =
        emitLocalDeclaration<MakeLabelOp>(getLocation(BB));
    }

    if (not Iterator->second.HasAssignLabel
        and Iterator->second.InsertPoint.isSet()) {
      mlir::OpBuilder::InsertionGuard Guard(Builder);
      restoreInsertionPointAfter(Builder, Iterator->second.InsertPoint);
      emitAssignLabel(Iterator->second.Label, getLocation(BB));
      Iterator->second.HasAssignLabel = true;
    }

    Builder.create<GoToOp>(Loc, Iterator->second.Label);
  }

  // Returns: { RequiresFullExpression, RequiresIndirection }
  std::pair<bool, bool> requiresFullExpression(const llvm::CallInst *Call) {
    if (Call->hasMetadata(PrototypeMDName)) {
      const auto *ModelCallType = getCallSitePrototype(Model, Call);
      auto Layout = abi::FunctionType::Layout::make(*ModelCallType);
      namespace ReturnMethod = abi::FunctionType::ReturnMethod;

      if (Layout.hasSPTAR())
        return { true, true };

      return { Layout.returnMethod() == ReturnMethod::RegisterSet, false };
    }

    return { llvm::isa<llvm::StructType>(Call->getType()), false };
  }

  // This function emits a single basic block as part of a larger C scope.
  // Nested scopes reached through branches at the end of this basic block are
  // emitted recursively using emitScope.
  RecursiveCoroutine<void>
  emitBasicBlock(const llvm::BasicBlock *BB,
                 const llvm::BasicBlock *InnerPostDom) {
    // Map BB to the MLIR block, emit label if necessary:
    {
      auto [Iterator, Inserted] = BlockMapping.try_emplace(BB);

      revng_assert(not Iterator->second.InsertPoint.isSet());
      Iterator->second.InsertPoint = saveInsertionPointAfter(Builder);

      if (not Inserted and Iterator->second.Label) {
        emitAssignLabel(Iterator->second.Label, getLocation(BB));
        Iterator->second.HasAssignLabel = true;
      }
    }

    const llvm::Instruction *Terminal = BB->getTerminator();
    bool HasGotoMarker = false;

    for (const llvm::Instruction &I : *BB) {
      if (&I == Terminal)
        break;

      if (auto *Alloca = llvm::dyn_cast<llvm::AllocaInst>(&I)) {
        const auto *Size = Alloca->getArraySize();

        // Non-constant alloca is not supported:
        revng_assert(not Size or llvm::isa<llvm::ConstantInt>(Size));

        clift::ValueType Type;
        if (hasStackTypeMetadata(Alloca)) {
          Type = importModelType(*getStackTypeFromMetadata(Alloca, Model));
        } else if (hasVariableTypeMetadata(Alloca)) {
          Type = importModelType(*getVariableTypeFromMetadata(Alloca, Model));
        } else {
          Type = importLLVMType(Alloca->getAllocatedType());

          if (Alloca->isArrayAllocation())
            Type = ArrayType::get(Context, Type, getConstantInt(Size));
        }

        auto Op = emitLocalDeclaration<LocalVariableOp>(getLocation(Alloca),
                                                        Type);
        auto [Iterator, Inserted] = AllocaMapping.try_emplace(Alloca, Op);
        revng_assert(Inserted);

        continue;
      }

      if (const llvm::CallInst *Call = llvm::dyn_cast<llvm::CallInst>(&I)) {
        if (isCallToTagged(Call, FunctionTags::GotoBlockMarker)) {
          HasGotoMarker = true;
          continue;
        }

        // Scope closer markers are just ignored. They only affect the scope
        // graph structure.
        if (isCallToTagged(Call, FunctionTags::ScopeCloserMarker))
          continue;

        // Some function calls are emitted in local variable initializers.
        if (auto [FE, Indirection] = requiresFullExpression(Call); FE) {
          mlir::Location Loc = getLocation(Call);

          auto Op = Builder.create<ExpressionStatementOp>(Loc);

          mlir::Value Local;
          emitExpressionTreeImpl(Op.getExpression(), [&]() {
            mlir::Value Initializer = emitExpression(Call, Loc);

            mlir::Type Type = Initializer.getType();
            Local = emitLocalDeclaration<LocalVariableOp>(Loc, Type);

            return Builder.create<AssignOp>(Loc, Type, Local, Initializer);
          });

          auto [Iterator, Inserted] = ValueMapping.try_emplace(Call, Local);
          revng_assert(Inserted);

          Iterator->second.Addressof = Indirection;
          continue;
        }
      }

      if (I.use_empty()) {
        // Any instruction with no uses can be considered the root of an
        // expression tree. The tree can be emitted in an expression statement.
        mlir::Location Loc = getLocation(&I);
        auto Op = Builder.create<ExpressionStatementOp>(Loc);
        emitExpressionTree(Op.getExpression(), &I, Loc);
      }

#if 0
      if (I.use_empty()) {
        // Any instruction with no uses can be considered the root of an
        // expression tree. The tree can be emitted in an expression statement.
        auto Op = Builder.create<ExpressionStatementOp>(mlir::UnknownLoc::get(Context));
        emitExpressionTree(&I, Op.getExpression());
      } else if (I.hasNUsesOrMore(2) or isUsedOutsideOfBlock(&I, BB)) {
        mlir::Region R;
        mlir::Type Type = emitExpressionTree(&I, R);

        // Any instruction with more than one use, or with a single use outside
        // of this block must be emitted into a local variable initializer.
        auto Op = Builder.create<LocalVariableOp>(mlir::UnknownLoc::get(Context),
                                                  Type,
                                                  "");

        // Move the local block into the initializer region.
        Op.getInitializer().push_back(R.getBlocks().remove(R.front()));

        // Map this instruction value to the newly created local variable.
        auto [It, Inserted] = ValueMapping.try_emplace(&I, Op);
        revng_assert(Inserted);
      }
#endif
    }

    if (Terminal->getNumSuccessors() == 1) {
      const llvm::BasicBlock *Succ = Terminal->getSuccessor(0);

      if (HasGotoMarker)
        emitGoto(getLocation(Terminal), Succ);
      else
        rc_recur emitScope(Succ, InnerPostDom);

      rc_return;
    }

    revng_assert(not HasGotoMarker);
    mlir::Location TerminalLoc = getLocation(Terminal);

    if (llvm::isa<llvm::UnreachableInst>(Terminal)) {
      auto Op = Builder.create<ExpressionStatementOp>(TerminalLoc);

      emitExpressionTreeImpl(Op.getExpression(), [&]() {
        return emitHelperCall(TerminalLoc,
                              "llvm.unreachable",
                              getVoidType(),
                              /*ParameterTypes=*/ {},
                              /*Arguments=*/{});
      });
    } else if (auto *Return = llvm::dyn_cast<llvm::ReturnInst>(Terminal)) {
      auto Op = Builder.create<ReturnOp>(TerminalLoc);
      if (const llvm::Value *Value = Return->getReturnValue()) {
        auto FunctionType = CurrentFunction.getCliftFunctionType();

        auto FuncReturnType = FunctionType.getReturnType();
        auto LLVMReturnType = FuncReturnType;

        if (CurrentLayout.hasSPTAR())
          LLVMReturnType = makePointerType(LLVMReturnType);

        emitExpressionTreeImpl(Op.getResult(), [&]() {
          mlir::Value ReturnValue = emitExpression(Value, TerminalLoc);

          if (auto AT = mlir::dyn_cast<ArrayType>(ReturnValue.getType())) {
            ReturnValue = emitCast(TerminalLoc,
                                   ReturnValue,
                                   makePointerType(AT.getElementType()),
                                   CastKind::Decay);

            ReturnValue = emitCast(TerminalLoc,
                                   ReturnValue,
                                   makePointerType(LLVMReturnType),
                                   CastKind::Bitcast);

            ReturnValue = Builder.create<IndirectionOp>(TerminalLoc,
                                                        LLVMReturnType,
                                                        ReturnValue);
          }

          ReturnValue = emitImplicitCast(TerminalLoc,
                                         ReturnValue,
                                         LLVMReturnType);

          if (CurrentLayout.hasSPTAR()) {
            ReturnValue = Builder.create<IndirectionOp>(TerminalLoc,
                                                        FuncReturnType,
                                                        ReturnValue);
          }

          return ReturnValue;
        });
      }
    } else if (auto *Branch = llvm::dyn_cast<llvm::BranchInst>(Terminal)) {
      auto Op = Builder.create<IfOp>(TerminalLoc);

      emitExpressionTree(Op.getCondition(),
                         Branch->getCondition(),
                         TerminalLoc);

      mlir::OpBuilder::InsertionGuard Guard(Builder);

      // Emit true branch:
      revng_assert(Op.getThen().empty());
      Builder.setInsertionPointToEnd(&Op.getThen().emplaceBlock());
      rc_recur emitScope(Branch->getSuccessor(0), InnerPostDom);

      // Emit false branch:
      revng_assert(Op.getElse().empty());
      Builder.setInsertionPointToEnd(&Op.getElse().emplaceBlock());
      rc_recur emitScope(Branch->getSuccessor(1), InnerPostDom);
    } else if (auto *Switch = llvm::dyn_cast<llvm::SwitchInst>(Terminal)) {
      llvm::SmallVector<uint64_t> CaseValues;
      CaseValues.reserve(Switch->getNumCases());

      for (auto CH : Switch->cases())
        CaseValues.push_back(CH.getCaseValue()->getZExtValue());

      auto Op = Builder.create<SwitchOp>(TerminalLoc, CaseValues);

      emitExpressionTree(Op.getCondition(),
                         Switch->getCondition(),
                         TerminalLoc);

      mlir::OpBuilder::InsertionGuard Guard(Builder);

      // Emit case blocks:
      for (auto [I, CH] : llvm::enumerate(Switch->cases())) {
        const llvm::BasicBlock *Succ = CH.getCaseSuccessor();

        revng_assert(Op.getCaseRegion(I).empty());
        Builder.setInsertionPointToEnd(&Op.getCaseRegion(I).emplaceBlock());

        if (hasBeenEmitted(Succ))
          emitGoto(TerminalLoc, Succ);
        else
          rc_recur emitScope(Succ, InnerPostDom);
      }

      // Emit default block:
      if (const llvm::BasicBlock *Succ = Switch->getDefaultDest();
          Succ != nullptr and isScopeGraphEdge(BB, Succ)) {

        revng_assert(Op.getDefaultCaseRegion().empty());
        Builder
          .setInsertionPointToEnd(&Op.getDefaultCaseRegion().emplaceBlock());

        if (hasBeenEmitted(Succ))
          emitGoto(TerminalLoc, Succ);
        else
          rc_recur emitScope(Succ, InnerPostDom);
      }
    } else {
      revng_abort("Unsupported terminal instruction");
    }
  }

  // This function emits a single logical C scope starting at BB and bounded by
  // OuterPostDom, which is not part of the scope. Any nested scopes are emitted
  // by emitBasicBlock.
  //
  // Putting it another way: this function goes down across a scope, while
  // emitBasicBlock recurses right, into nested scopes:
  //
  // emitScope
  //     |
  //     +--> emitBasicBlock --> emitScope
  //                                 |
  //                                 + --> emitBasicBlock
  //                                 |
  //                                 + --> emitBasicBlock
  //                                 !
  //     |
  //     +--> emitBasicBlock
  //     !
  //
  // The subgraph of basic blocks recursively emitted by emitBasicBlock starting
  // at some basic block B is bounded by the immediate post dominator of B,
  // which is itself always equal to or dominated by OuterPostDom.
  RecursiveCoroutine<void> emitScope(const llvm::BasicBlock *BB,
                                     const llvm::BasicBlock *OuterPostDom) {
    while (BB != OuterPostDom) {
      const auto *InnerPostDom = PostDomTree[BB]->getIDom()->getBlock();
      revng_assert(PostDomTree.dominates(OuterPostDom, InnerPostDom));

      rc_recur emitBasicBlock(BB, InnerPostDom);
      BB = InnerPostDom;
    }
  }

  /* LLVM module import */

  static llvm::ArrayRef<abi::FunctionType::Layout::Argument>
  getLayoutArguments(const abi::FunctionType::Layout &Layout) {
    llvm::ArrayRef LayoutArguments = Layout.Arguments;
    if (Layout.hasSPTAR())
      LayoutArguments = LayoutArguments.drop_front();
    return LayoutArguments;
  }

  void importFunction(const llvm::Function *F) {
    if (F->getName() == "llvm.unreachable")
      revng_abort("Encountered function with name llvm.unreachable");

    const model::Function *MF = llvmToModelFunction(Model, *F);
    if (MF == nullptr)
      return;

    auto Op = getOrEmitSymbol(F, [&]() -> FunctionOp {
      return emitModelFunctionDeclaration(F, MF);
    });

    revng_assert(Op.getArgumentTypes().size() == F->arg_size());

    revng_assert(Op.getBody().empty());
    mlir::Block &BodyBlock = Op.getBody().emplaceBlock();

    CurrentFunction = Op;
    CurrentLayout = abi::FunctionType::Layout::make(*MF->Prototype());

    LocalDeclarationInsertPoint = makeInsertionPointAfterStart(&BodyBlock);

    // WIP: Get rid of label and local names
    LabelCount = 0;
    LocalCount = 0;

    // Clear the mappings once this function is emitted.
    auto MappingGuard = llvm::make_scope_exit([&]() {
      ArgumentMapping.clear();
      BlockMapping.clear();
      AllocaMapping.clear();
      ValueMapping.clear();
    });

    llvm::ArrayRef LayoutArguments = getLayoutArguments(CurrentLayout);
    revng_assert(F->arg_size() == Op.getArgumentTypes().size());
    revng_assert(F->arg_size() == LayoutArguments.size());

    for (const auto [A, T, L] : llvm::zip(F->args(),
                                          Op.getArgumentTypes(),
                                          LayoutArguments)) {
      ArgumentMapping.try_emplace(&A,
                                  BodyBlock.addArgument(T, Op->getLoc()),
                                  /*IsByAddress=*/isByAddressParameter(L),
                                  /*CastType=*/importLLVMType(A.getType()));
    }

    PostDomTree.recalculate(const_cast<llvm::Function &>(*F));

    mlir::OpBuilder::InsertionGuard BuilderGuard(Builder);
    Builder.setInsertionPointToEnd(&BodyBlock);

    emitScope(&F->getEntryBlock(), /*OuterPostDom=*/nullptr);
  }

  void importModule(mlir::ModuleOp ModuleOp, const llvm::Module *Module) {
    CurrentModule = ModuleOp;
    Builder.setInsertionPointToEnd(ModuleOp.getBody());

    for (const llvm::Function &F : Module->functions())
      importFunction(&F);
  }

  mlir::MLIRContext *const Context;
  const model::Binary &Model;
  mlir::OpBuilder Builder;

  mlir::ModuleOp CurrentModule;
  clift::FunctionOp CurrentFunction;
  abi::FunctionType::Layout CurrentLayout;

  mlir::OpBuilder::InsertPoint LocalDeclarationInsertPoint;

  ScopeGraphPostDomTree PostDomTree;

  clift::ValueType VoidTypeCache;

  uint64_t PointerSize;
  clift::ValueType PointerTypeCache;
  clift::ValueType IntptrTypeCache;

  struct ArgumentMappingInfo {
    mlir::Value Argument;

    // True if the argument is a by-address argument in the LLVM IR and should
    // be dereferenced before use in Clift.
    bool IsByAddress;

    mlir::Type CastType;
  };

  struct BlockMappingInfo {
    // Point where the mapped LLVM IR basic block is emitted. If the block has
    // not yet been visited, this is not set. Note that the LLVM IR basic block
    // is not necessarily emitted at the *start* of any MLIR block.
    mlir::OpBuilder::InsertPoint InsertPoint;

    // The result value of the MakeLabelOp to be used as the target for gotos
    // jumping into the mapped LLVM IR basic block.
    mlir::Value Label;

    // True if the AssignLabelOp has already been created.
    bool HasAssignLabel = false;
  };

  struct ValueMappingInfo {
    mlir::Value Value;

    // True if the address of the value should be taken before use in Clift.
    bool Addressof;

    ValueMappingInfo(mlir::Value Value) : Value(Value), Addressof(false) {}
  };

  // Maps LLVM object to a Clift global op.
  llvm::DenseMap<const llvm::GlobalObject *, clift::GlobalOpInterface>
    SymbolMapping;

  llvm::DenseMap<const llvm::GlobalVariable *, StringLiteral> StringMapping;

  // Maps helper name to Clift function op.
  llvm::DenseMap<llvm::StringRef, clift::FunctionOp> HelperMapping;

  // Maps LLVM IR function argument to a Clift argument description.
  llvm::DenseMap<const llvm::Argument *, ArgumentMappingInfo> ArgumentMapping;

  // Maps LLVM IR basic block to a Clift import state for that block.
  llvm::DenseMap<const llvm::BasicBlock *, BlockMappingInfo> BlockMapping;

  // Maps LLVM IR alloca to a Clift local variable op result.
  llvm::DenseMap<const llvm::AllocaInst *, mlir::Value> AllocaMapping;

  // Maps an arbitrary LLVM IR value to a Clift value description.
  llvm::DenseMap<const llvm::Value *, ValueMappingInfo> ValueMapping;

  unsigned LabelCount = 0;
  unsigned LocalCount = 0;
};

} // namespace

void clift::importLLVM(mlir::ModuleOp ModuleOp,
                       const model::Binary &Model,
                       const llvm::Module *Module) {
  revng_assert(clift::hasModuleAttr(ModuleOp));
  LLVMCodeImporter::import(ModuleOp, Model, Module);
}

mlir::OwningOpRef<mlir::ModuleOp>
clift::importLLVM(mlir::MLIRContext *Context,
                  const model::Binary &Model,
                  const llvm::Module *Module) {
  // WIP: Set location from module LLVM source file and name.
  auto Op = createOperation<mlir::ModuleOp>(Context,
                                            mlir::UnknownLoc::get(Context));

  clift::setModuleAttr(Op.get());
  importLLVM(Op.get(), Model, Module);

  return Op;
}
