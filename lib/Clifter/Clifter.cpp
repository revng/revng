//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/Support/GenericDomTree.h"

#include "revng/ABI/FunctionType/Layout.h"
#include "revng/ADT/RecursiveCoroutine.h"
#include "revng/ADT/ScopedExchange.h"
#include "revng/Clift/CliftDialect.h"
#include "revng/CliftImportModel/ImportModel.h"
#include "revng/Clifter/Clifter.h"
#include "revng/LocalVariables/LocalVariableHelpers.h"
#include "revng/Model/Binary.h"
#include "revng/Model/IRHelpers.h"
#include "revng/Pipeline/Location.h"
#include "revng/Pipes/Ranks.h"
#include "revng/RestructureCFG/ScopeGraphGraphTraits.h"
#include "revng/Support/Debug.h"
#include "revng/Support/FunctionTags.h"
#include "revng/Support/Identifier.h"

namespace clift = mlir::clift;
using namespace clift;

namespace {

static Logger<> BasicBlockLog{ "llvm-to-clift-basic-blocks" };
static Logger<> ExpressionLog{ "llvm-to-clift-expressions" };

// These helper functions are used to save insertion points *after* an
// operation, while the usual operating mode of `OpBuilder::InsertPoint` is
// *before* and operation. On save, the saved iterator is moved backwards by one
// position, conversely moved forward on restore. For the begin/end positions,
// it can be imagined that they loop around to the other. For this reason the
// end position actually signifies insertion at the start of the block.
//
// The scenario where saving *after* vs *before* an operation (or end) makes a
// difference is as follows: We emit some operation X in block B and want to
// save the position *after* X. If X is at the end B, saving the position
// normally would result in saving an end iterator for B. If some new operation
// Y is subsequently inserted at the end of B, the meaning of the previously
// saved position is changed. The iterator still points to the end of B, but now
// the preceding operation (and so the operation *after* which we insert) is no
// longer X but rather Y. For the same reason the usual mechanism does not allow
// saving the position at the beginning of a block (or *after* an imagined
// operation before the first operation in the block, if any).

[[nodiscard]] static mlir::OpBuilder::InsertPoint
saveInsertionPointAfter(const mlir::OpBuilder &Builder) {
  mlir::Block *Block = Builder.getInsertionBlock();
  mlir::Block::iterator Point = Builder.getInsertionPoint();

  // End iterator specifies insertion at the start of Block. Non-end iterator
  // specifies insertion *after* the operation referred to by the iterator.
  Point = Point == Block->begin() ? Block->end() : std::prev(Point);

  return mlir::OpBuilder::InsertPoint(Block, Point);
}

static void
restoreInsertionPointAfter(mlir::OpBuilder &Builder,
                           mlir::OpBuilder::InsertPoint InsertPoint) {
  revng_assert(InsertPoint.isSet());

  mlir::Block *Block = InsertPoint.getBlock();
  mlir::Block::iterator Point = InsertPoint.getPoint();

  // Convert an end iterator back into begin and advance non-end iterators. This
  // is because the builder inserts operations *before* the specified iterator.
  Point = Point == Block->end() ? Block->begin() : std::next(Point);

  Builder.setInsertionPoint(Block, Point);
}

using ScopeGraphPostDomTree = llvm::PostDomTreeOnView<llvm::BasicBlock, Scope>;

class ClifterImpl final : public Clifter {
public:
  // It is important only to query minimal properties about the model in the
  // importer constructor to avoid all imported functions depending on those
  // properties.
  explicit ClifterImpl(mlir::ModuleOp Module, const model::Binary &Model) :
    Context(Module.getContext()),
    CurrentModule(Module),
    Model(Model),
    Builder(Context),
    PointerSize(getPointerSize(Model.Architecture())) {
    revng_assert(clift::hasModuleAttr(Module));
    Builder.setInsertionPointToEnd(Module.getBody());
  }

private:
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
    if (IntegerWidth == 1)
      IntegerWidth = 8;

    revng_check(IntegerWidth % 8 == 0);

    // Convert the size in bits to a size in octets:
    return IntegerWidth / 8;
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
    if (not VoidPointerTypeCache)
      VoidPointerTypeCache = makePointerType(getVoidType());
    return VoidPointerTypeCache;
  }

  clift::ValueType getIntptrType() {
    if (not IntptrTypeCache)
      IntptrTypeCache = getPrimitiveType(PointerSize);
    return IntptrTypeCache;
  }

  template<typename FunctionT>
  model::UpcastableType getPrototype(const FunctionT &Function) {
    if (auto &&Type = Function.Prototype())
      return Type;
    return Model.makeType(Model.defaultPrototype()->key());
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

  uint64_t getConstantInt(const llvm::Value *Value) {
    return llvm::cast<llvm::ConstantInt>(Value)->getZExtValue();
  }

  // Determines if a argument is passed as a pointer to the actual value. In
  // these cases an indirection must be inserted when using the argument.
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

  template<typename EmitT>
  auto getOrEmitSymbol(const llvm::GlobalObject *G, EmitT &&Emit) {
    using ReturnType = std::invoke_result_t<EmitT>;
    auto [Iterator, Inserted] = SymbolMapping.try_emplace(G);
    if (Inserted) {
      mlir::OpBuilder::InsertionGuard Guard(Builder);
      Builder.setInsertionPointToEnd(CurrentModule.getBody());
      Iterator->second = std::forward<EmitT>(Emit)();
    }
    return mlir::cast<ReturnType>(Iterator->second);
  }

  static std::string getHelperStructFieldName(unsigned Index) {
    std::string Name;
    {
      llvm::raw_string_ostream Out(Name);
      Out << "field_" << Index;
    }
    return Name;
  }

  // Import an LLVM type used as a helper function return type. If the type is a
  // struct type, create a Clift struct type with a handle derived from the
  // helper function name.
  clift::ValueType importHelperReturnType(const llvm::Type *Type,
                                          llvm::StringRef HelperName) {
    auto Struct = llvm::dyn_cast<llvm::StructType>(Type);

    if (not Struct)
      return importLLVMType(Type);

    revng_assert(Struct->isLiteral());

    auto Location = pipeline::location(revng::ranks::HelperStructType,
                                       HelperName.str());

    llvm::SmallVector<FieldAttr> Fields;
    Fields.reserve(Struct->getNumElements());

    uint64_t Offset = 0;
    for (auto [I, T] : llvm::enumerate(Struct->elements())) {
      clift::ValueType FieldType = importLLVMType(T);

      std::string FieldName = getHelperStructFieldName(I);
      std::string FieldHandle = Location
                                  .extend(revng::ranks::HelperStructField,
                                          FieldName)
                                  .toString();

      Fields.push_back(FieldAttr::get(Context,
                                      FieldHandle,
                                      makeNameAttr<FieldAttr>(Context,
                                                              FieldHandle,
                                                              FieldName),
                                      Offset,
                                      FieldType));
      Offset += FieldType.getByteSize();
    }

    auto Handle = pipeline::locationString(revng::ranks::HelperStructType,
                                           HelperName.str());

    std::string
      Name = Model.Configuration().Naming().artificialReturnValuePrefix().str()
             + sanitizeIdentifier(HelperName);

    auto Definition = StructAttr::get(Context,
                                      Handle,
                                      makeNameAttr<StructAttr>(Context,
                                                               Handle,
                                                               Name),
                                      Offset,
                                      Fields);

    return StructType::get(Context, Definition);
  }

  // Import a Clift function type from an LLVM function type and a helper name
  // used for deriving the type handle.
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

    return FunctionType::get(Context,
                             Handle,
                             makeNameAttr<FunctionType>(Context, Handle),
                             ReturnType,
                             ParameterTypes);
  }

  // Import a Clift function type from an LLVM function, using the name of the
  // function for deriving the type handle.
  clift::FunctionType importHelperType(const llvm::Function *F) {
    return importHelperType(F->getFunctionType(), F->getName());
  }

  clift::FunctionOp
  emitHelperFunctionDeclaration(mlir::Location Loc,
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
    } else {
      revng_assert(Iterator->second.getCliftFunctionType() == FunctionType);
    }

    return Iterator->second;
  }

  clift::FunctionOp emitHelperFunctionDeclaration(const llvm::Function *F) {
    return getOrEmitSymbol(F, [&]() -> clift::FunctionOp {
      return emitHelperFunctionDeclaration(getLocation(F->getSubprogram()),
                                           F->getName(),
                                           importHelperType(F));
    });
  }

  template<typename FunctionT, typename RankT, typename... ArgsT>
  clift::FunctionOp emitModelFunctionDeclaration(const llvm::Function *F,
                                                 const FunctionT &MF,
                                                 const RankT &Rank,
                                                 ArgsT &&...Args) {
    model::UpcastableType Prototype = getPrototype(MF);
    auto FunctionType = importModelType<clift::FunctionType>(*Prototype);

    return getOrEmitSymbol(F, [&]() -> clift::FunctionOp {
      // It is important not to query any model properties in this scope, as
      // doing so would break invalidation when the import of a function uses a
      // declaration already emitted during the import of previous function.
      auto Op = Builder.create<FunctionOp>(getLocation(F->getSubprogram()),
                                           F->getName(),
                                           FunctionType);
      Op.setHandle(pipeline::locationString(Rank,
                                            std::forward<ArgsT>(Args)...));
      return Op;
    });
  }

  clift::FunctionOp emitIsolatedFunctionDeclaration(const llvm::Function *F,
                                                    const model::Function *MF) {
    return emitModelFunctionDeclaration(F,
                                        *MF,
                                        revng::ranks::Function,
                                        MF->Entry());
  }

  clift::FunctionOp emitImportedFunctionDeclaration(const llvm::Function *F) {
    llvm::StringRef Name = F->getName();
    revng_check(Name.consume_front("dynamic_"));

    auto It = Model.ImportedDynamicFunctions().find(Name.str());
    revng_check(It != Model.ImportedDynamicFunctions().end());

    return emitModelFunctionDeclaration(F,
                                        *It,
                                        revng::ranks::DynamicFunction,
                                        Name.str());
  }

  clift::FunctionOp emitFunctionDeclaration(const llvm::Function *F) {
    if (const model::Function *MF = llvmToModelFunction(Model, *F))
      return emitIsolatedFunctionDeclaration(F, MF);

    auto Tags = FunctionTags::TagsSet::from(F);
    if (Tags.contains(FunctionTags::DynamicFunction))
      return emitImportedFunctionDeclaration(F);

    return emitHelperFunctionDeclaration(F);
  }

  clift::ValueType emitGlobalObject(const llvm::GlobalObject *G) {
    if (const auto *F = llvm::dyn_cast<llvm::Function>(G))
      return emitFunctionDeclaration(F).getCliftFunctionType();

    revng_abort("Unsupported global object kind");
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

  // Apply an implicit conversion to the requested type. The conversion is only
  // applied in the case that neither the source nor target values have array
  // type, and as long as both have the same size. If the cast is possible, it
  // is naturally performed using the bitcast mode. If the cast is not possible,
  // no cast is emitted and the original value is returned instead. In the case
  // of failure to convert LLVM IR to Clift, this is a likely point of failure.
  // When that happens, invalid Clift is emitted, causing subsequent failure in
  // verification.
  //
  // During development of the LLVM IR to Clift converter, this behaviour was
  // found to be more useful for debugging than simply aborting. It is generally
  // easier to find the problem when full context of the surrounding Clift code
  // is available.
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

  // Used for emitting operations acting on integer operands (arithmetic,
  // comparison, etc.) of a specific kind. Before applying the operation, the
  // operands are automatically converted to the requested kind, if necessary.
  // After the operation, the result is automaticlaly converted to generic kind.
  mlir::Value emitIntegerOp(mlir::Location Loc,
                            PrimitiveKind Kind,
                            auto ApplyOperation,
                            std::same_as<mlir::Value> auto... Operands) {
    auto ConvertToKind = [&](mlir::Value &Value, PrimitiveKind Kind) {
      auto UnderlyingType = getUnderlyingIntegerType(Value.getType());
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

  // Applies an integer cast of the operand to the specified size, using the
  // specified primitive kind. Finally the result is converted to generic kind.
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
    return emitHelperFunctionDeclaration(mlir::UnknownLoc::get(Context),
                                         HelperName,
                                         FunctionType);
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
    return emitHelperCall(Loc, emitHelperFunctionDeclaration(F), Arguments);
  }

  mlir::Value emitHelperCall(mlir::Location Loc,
                             llvm::StringRef HelperName,
                             mlir::Type ReturnType,
                             llvm::ArrayRef<mlir::Type> ParameterTypes,
                             llvm::ArrayRef<mlir::Value> Arguments) {
    auto Handle = pipeline::locationString(revng::ranks::HelperFunction,
                                           HelperName.str());

    auto NameAttr = makeNameAttr<clift::FunctionType>(Context, Handle);
    auto FunctionType = clift::FunctionType::get(Context,
                                                 Handle,
                                                 NameAttr,
                                                 ReturnType,
                                                 ParameterTypes);

    return emitHelperCall(Loc,
                          getHelperFunction(HelperName, FunctionType),
                          Arguments);
  }

  RecursiveCoroutine<mlir::Value>
  emitStructInitializer(const llvm::CallInst *Call) {
    mlir::Location Loc = getLocation(Call);

    auto UseBegin = Call->use_begin();
    auto UseEnd = Call->use_end();
    revng_assert(UseBegin != UseEnd);

    // StructInitializer must have exactly one user of type ReturnInst.
    revng_assert(llvm::isa<llvm::ReturnInst>(UseBegin->getUser()));
    revng_assert(std::next(UseBegin) == UseEnd);

    auto T = mlir::dyn_cast<StructType>(CurrentFunction.getCliftReturnType());
    revng_assert(Call->arg_size() == T.getFields().size());

    llvm::SmallVector<mlir::Value> Initializers;
    for (const auto &[A, F] : llvm::zip(Call->args(), T.getFields())) {
      mlir::Value Value = rc_recur emitExpression(A, Loc);
      Value = emitImplicitCast(Loc, Value, F.getType());
      Initializers.push_back(Value);
    }

    rc_return Builder.create<AggregateOp>(Loc, T, Initializers);
  }

  RecursiveCoroutine<mlir::Value>
  emitOpaqueExtractValue(const llvm::CallInst *Call) {
    mlir::Location Loc = getLocation(Call);

    mlir::Value Aggregate = rc_recur emitExpression(Call->getArgOperand(0),
                                                    Loc);

    uint64_t Index = getConstantInt(Call->getArgOperand(1));
    auto Struct = mlir::cast<StructType>(Aggregate.getType());
    revng_assert(Index < Struct.getFields().size());

    mlir::Type FieldType = Struct.getFields()[Index].getType();
    mlir::Value FieldValue = Builder.create<AccessOp>(Loc,
                                                      FieldType,
                                                      Aggregate,
                                                      /*indirect=*/false,
                                                      Index);

    rc_return emitImplicitCast(Loc,
                               FieldValue,
                               importLLVMType(Call->getType()));
  }

  RecursiveCoroutine<mlir::Value> emitHelperCall(const llvm::CallInst *Call) {
    // TODO: Use getCalledFunction. Pending fix for uniqued-by-prototype issue.
    // const llvm::Function *Callee = Call->getCalledFunction();
    auto *Callee = llvm::dyn_cast<llvm::Function>(Call->getCalledOperand());
    revng_assert(Callee != nullptr);

    auto Tags = FunctionTags::TagsSet::from(Callee);

    if (Tags.contains(FunctionTags::StructInitializer))
      rc_return rc_recur emitStructInitializer(Call);

    if (Tags.contains(FunctionTags::OpaqueExtractValue))
      rc_return rc_recur emitOpaqueExtractValue(Call);

    mlir::Location Loc = getLocation(Call);

    llvm::SmallVector<mlir::Value> Arguments;
    Arguments.reserve(Call->arg_size());

    for (const llvm::Value *Argument : Call->args())
      Arguments.push_back(rc_recur emitExpression(Argument, Loc));

    rc_return emitHelperCall(Loc,
                             emitHelperFunctionDeclaration(Callee),
                             Arguments);
  }

  struct StringLiteral {
    clift::ArrayType Type;
    std::string Data;
  };

  bool detectStringLiteralImpl(const llvm::GlobalVariable *V,
                               StringLiteral &String) {
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

  // Returns a non-null pointer to a string literal description if the specified
  // global variable represents a string literal. Successful detections are
  // cached in a table.
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
    LoggerIndent Indent(ExpressionLog);

    if (auto G = llvm::dyn_cast<llvm::GlobalObject>(V)) {
      if (const StringLiteral *String = detectStringLiteral(G)) {
        revng_log(ExpressionLog, "llvm::GlobalObject -> StringOp");
        auto Op = Builder.create<StringOp>(SurroundingLocation,
                                           String->Type,
                                           std::move(String->Data));

        auto Type = makePointerType(String->Type.getElementType());
        rc_return emitCast(SurroundingLocation, Op, Type, CastKind::Decay);
      }

      if (const auto *F = llvm::dyn_cast<llvm::Function>(G)) {
        revng_log(ExpressionLog, "llvm::Function -> UseOp");
        auto Function = emitFunctionDeclaration(F);
        rc_return Builder.create<UseOp>(SurroundingLocation,
                                        Function.getCliftFunctionType(),
                                        Function.getSymNameAttr());
      }

      revng_abort("Unsupported global object kind");
    }

    if (auto It = ValueMapping.find(V); It != ValueMapping.end()) {
      revng_log(ExpressionLog, "<existing value>");

      mlir::Value Value = It->second.Value;

      if (It->second.TakeAddressOnUse) {
        Value = Builder.create<AddressofOp>(SurroundingLocation,
                                            makePointerType(Value.getType()),
                                            Value);
      }

      rc_return Value;
    }

    if (auto A = llvm::dyn_cast<llvm::Argument>(V)) {
      revng_log(ExpressionLog, "llvm::Argument");
      LoggerIndent Indent(ExpressionLog);

      auto It = ArgumentMapping.find(A);
      revng_assert(It != ArgumentMapping.end());

      mlir::Value Value = It->second.Argument;

      if (It->second.TakeAddressOnUse) {
        Value = Builder.create<AddressofOp>(SurroundingLocation,
                                            makePointerType(Value.getType()),
                                            Value);
      }

      rc_return emitImplicitCast(SurroundingLocation,
                                 Value,
                                 It->second.CastType);
    }

    if (auto U = llvm::dyn_cast<llvm::UndefValue>(V)) {
      revng_log(ExpressionLog, "llvm::UndefValue -> UndefOp");
      mlir::Type Type = importLLVMType(U->getType());
      rc_return Builder.create<UndefOp>(SurroundingLocation, Type);
    }

    if (auto C = llvm::dyn_cast<llvm::ConstantInt>(V)) {
      revng_log(ExpressionLog, "llvm::ConstantInt -> ImmediateOp");
      const llvm::IntegerType *T = llvm::cast<llvm::IntegerType>(C->getType());
      rc_return Builder.create<ImmediateOp>(SurroundingLocation,
                                            importLLVMIntegerType(T),
                                            C->getZExtValue());
    }

    if (auto N = llvm::dyn_cast<llvm::ConstantPointerNull>(V)) {
      revng_log(ExpressionLog, "llvm::ConstantPointerNull -> ImmediateOp");
      auto Op = Builder.create<ImmediateOp>(SurroundingLocation,
                                            getIntptrType(),
                                            /*Value=*/0);

      LoggerIndent Indent(ExpressionLog);
      rc_return emitCast(SurroundingLocation, Op, getVoidPointerType());
    }

    if (auto E = llvm::dyn_cast<llvm::ConstantExpr>(V)) {
      revng_log(ExpressionLog, "llvm::ConstantExpr");
      LoggerIndent Indent(ExpressionLog);

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
        revng_log(ExpressionLog, "llvm::AllocaInst");
        LoggerIndent Indent(ExpressionLog);

        auto Type = makePointerType(It->second.getType());
        auto Value = Builder.create<AddressofOp>(Loc, Type, It->second);
        rc_return emitImplicitCast(Loc, Value, importLLVMType(I->getType()));
      }

      revng_assert(not llvm::isa<llvm::Constant>(I->getArraySize()));
      revng_abort("Non-constant alloca is not supported.");
    }

    if (auto I = llvm::dyn_cast<llvm::LoadInst>(V)) {
      revng_log(ExpressionLog, "llvm::LoadInst");
      LoggerIndent Indent(ExpressionLog);

      mlir::Location Loc = getLocation(I);

      revng_log(ExpressionLog, "Subexpression:");
      mlir::Value Pointer = (LoggerIndent(ExpressionLog),
                             rc_recur emitExpression(I->getPointerOperand(),
                                                     Loc));

      clift::ValueType ValueType = importLLVMType(V->getType());
      clift::ValueType PointerType = makePointerType(ValueType);

      Pointer = Builder.create<CastOp>(Loc,
                                       PointerType,
                                       Pointer,
                                       CastKind::Bitcast);

      revng_log(ExpressionLog, "IndirectionOp");
      rc_return Builder.create<IndirectionOp>(Loc, ValueType, Pointer);
    }

    if (auto I = llvm::dyn_cast<llvm::StoreInst>(V)) {
      revng_log(ExpressionLog, "llvm::StoreInst");
      LoggerIndent Indent(ExpressionLog);

      mlir::Location Loc = getLocation(I);

      revng_log(ExpressionLog, "Pointer subexpression:");
      mlir::Value Pointer = (LoggerIndent(ExpressionLog),
                             rc_recur emitExpression(I->getPointerOperand(),
                                                     Loc));

      revng_log(ExpressionLog, "Value subexpression:");
      mlir::Value Value = (LoggerIndent(ExpressionLog),
                           rc_recur emitExpression(I->getValueOperand(), Loc));

      Pointer = Builder.create<CastOp>(Loc,
                                       makePointerType(Value.getType()),
                                       Pointer,
                                       CastKind::Bitcast);

      revng_log(ExpressionLog, "IndirectionOp");
      mlir::Value Assignee = Builder.create<IndirectionOp>(Loc,
                                                           Value.getType(),
                                                           Pointer);

      revng_log(ExpressionLog, "AssignOp");
      rc_return Builder.create<AssignOp>(Loc, Value.getType(), Assignee, Value);
    }

    if (auto I = llvm::dyn_cast<llvm::BinaryOperator>(V)) {
      revng_log(ExpressionLog, "llvm::BinaryOperator");
      LoggerIndent Indent(ExpressionLog);

      using Operators = llvm::BinaryOperator::BinaryOps;

      mlir::Location Loc = getLocation(I);

      revng_log(ExpressionLog, "LHS subexpression");
      mlir::Value Lhs = (LoggerIndent(ExpressionLog),
                         rc_recur emitExpression(I->getOperand(0), Loc));

      revng_log(ExpressionLog, "RHS subexpression");
      mlir::Value Rhs = (LoggerIndent(ExpressionLog),
                         rc_recur emitExpression(I->getOperand(1), Loc));

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
          revng_log(ExpressionLog, "AddOp");
          return emitExpr<AddOp>(Loc, Type, Lhs, Rhs);
        case Operators::Sub:
          revng_log(ExpressionLog, "SubOp");
          return emitExpr<SubOp>(Loc, Type, Lhs, Rhs);
        case Operators::Mul:
          revng_log(ExpressionLog, "MulOp");
          return emitExpr<MulOp>(Loc, Type, Lhs, Rhs);
        case Operators::SDiv:
        case Operators::UDiv:
          revng_log(ExpressionLog, "DivOp");
          return emitExpr<DivOp>(Loc, Type, Lhs, Rhs);
        case Operators::SRem:
        case Operators::URem:
          revng_log(ExpressionLog, "RemOp");
          return emitExpr<RemOp>(Loc, Type, Lhs, Rhs);
        case Operators::Shl:
          revng_log(ExpressionLog, "ShiftLeftOp");
          return emitExpr<ShiftLeftOp>(Loc, Type, Lhs, Rhs);
        case Operators::LShr:
        case Operators::AShr:
          revng_log(ExpressionLog, "ShiftRightOp");
          return emitExpr<ShiftRightOp>(Loc, Type, Lhs, Rhs);
        case Operators::And:
          revng_log(ExpressionLog, "BitwiseAndOp");
          return emitExpr<BitwiseAndOp>(Loc, Type, Lhs, Rhs);
        case Operators::Or:
          revng_log(ExpressionLog, "BitwiseOrOp");
          return emitExpr<BitwiseOrOp>(Loc, Type, Lhs, Rhs);
        case Operators::Xor:
          revng_log(ExpressionLog, "BitwiseXorOp");
          return emitExpr<BitwiseXorOp>(Loc, Type, Lhs, Rhs);
        default:
          revng_abort("Unsupported LLVM binary operator.");
        }
      };

      rc_return emitIntegerOp(Loc, Kind, EmitOp, Lhs, Rhs);
    }

    if (auto I = llvm::dyn_cast<llvm::ICmpInst>(V)) {
      revng_log(ExpressionLog, "llvm::ICmpInst");
      LoggerIndent Indent(ExpressionLog);

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

      revng_log(ExpressionLog, "LHS subexpression");
      mlir::Value Lhs = (LoggerIndent(ExpressionLog),
                         rc_recur emitExpression(I->getOperand(0), Loc));

      revng_log(ExpressionLog, "RHS subexpression");
      mlir::Value Rhs = (LoggerIndent(ExpressionLog),
                         rc_recur emitExpression(I->getOperand(1), Loc));

      auto *IntegerType = llvm::cast<llvm::IntegerType>(V->getType());
      auto Type = importLLVMIntegerType(IntegerType, PrimitiveKind::SignedKind);

      auto EmitOp = [&](mlir::Value Lhs, mlir::Value Rhs) {
        switch (I->getPredicate()) {
        case ICMP_EQ:
          revng_log(ExpressionLog, "CmpEqOp");
          return emitExpr<CmpEqOp>(Loc, Type, Lhs, Rhs);
        case ICMP_NE:
          revng_log(ExpressionLog, "CmpNeOp");
          return emitExpr<CmpNeOp>(Loc, Type, Lhs, Rhs);
        case ICMP_SGT:
        case ICMP_UGT:
          revng_log(ExpressionLog, "CmpGtOp");
          return emitExpr<CmpGtOp>(Loc, Type, Lhs, Rhs);
        case ICMP_SGE:
        case ICMP_UGE:
          revng_log(ExpressionLog, "CmpGeOp");
          return emitExpr<CmpGeOp>(Loc, Type, Lhs, Rhs);
        case ICMP_SLT:
        case ICMP_ULT:
          revng_log(ExpressionLog, "CmpLtOp");
          return emitExpr<CmpLtOp>(Loc, Type, Lhs, Rhs);
        case ICMP_SLE:
        case ICMP_ULE:
          revng_log(ExpressionLog, "CmpLeOp");
          return emitExpr<CmpLeOp>(Loc, Type, Lhs, Rhs);
        default:
          revng_abort("Unsupported LLVM comparison predicate.");
        }
      };

      rc_return emitIntegerOp(Loc, Kind, EmitOp, Lhs, Rhs);
    }

    if (auto I = llvm::dyn_cast<llvm::CastInst>(V)) {
      revng_log(ExpressionLog, "llvm::CastInst");
      LoggerIndent Indent(ExpressionLog);

      mlir::Location Loc = getLocation(I);

      revng_log(ExpressionLog, "Subexpression:");
      mlir::Value Operand = (LoggerIndent(ExpressionLog),
                             rc_recur emitExpression(I->getOperand(0), Loc));

      auto getIntegerSize = [](const llvm::Type *Type) {
        const auto *IntegerType = llvm::cast<llvm::IntegerType>(Type);
        return ClifterImpl::getIntegerSize(IntegerType->getBitWidth());
      };

      auto EmitIntegerCast = [&](PrimitiveKind Kind) {
        return emitIntegerCast(Loc,
                               Operand,
                               getIntegerSize(V->getType()),
                               Kind);
      };

      switch (I->getOpcode()) {
        using Operators = llvm::CastInst::CastOps;
      case Operators::Trunc:
        rc_return EmitIntegerCast(PrimitiveKind::GenericKind);
      case Operators::SExt:
        rc_return EmitIntegerCast(PrimitiveKind::SignedKind);
      case Operators::ZExt:
        rc_return EmitIntegerCast(PrimitiveKind::UnsignedKind);
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
      revng_log(ExpressionLog, "llvm::CallInst");
      LoggerIndent Indent(ExpressionLog);

      mlir::Location Loc = getLocation(I);

      // Any call without a prototype is considered a helper function and is
      // imported separately:
      if (not I->hasMetadata(PrototypeMDName))
        rc_return emitHelperCall(I);

      const auto *ModelCallType = getCallSitePrototype(Model, I);
      auto Layout = abi::FunctionType::Layout::make(*ModelCallType);

      auto CallType = importModelType<clift::FunctionType>(*ModelCallType);

      revng_log(ExpressionLog, "Callee subexpression:");
      mlir::Value Function = (LoggerIndent(ExpressionLog),
                              rc_recur emitExpression(I->getCalledOperand(),
                                                      Loc));

      clift::FunctionType
        FunctionType = getFunctionOrFunctionPointerFunctionType(Function
                                                                  .getType());

      // If the call type does not match the function type of the callee,
      // the callee must first be converted to a pointer to the appropriate
      // function type:
      if (CallType != FunctionType) {
        // If the callee is a function and not a pointer to function, it must
        // be decayed to a pointer before applying the type conversion:
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

      // Process each argument subexpression and apply the required conversions:
      llvm::SmallVector<mlir::Value> Arguments;
      for (size_t J = 0;
           auto [A, T, L] :
           llvm::zip(I->args(), CallType.getArgumentTypes(), LayoutArguments)) {

        revng_log(ExpressionLog, "Argument " << J++ << " subexpression:");
        mlir::Value Argument = (LoggerIndent(ExpressionLog),
                                rc_recur emitExpression(A, Loc));

        // Parameters which are passed by address in the LLVM IR are passed by
        // value in Clift, and so the LLVM IR pointer value must be dereferenced
        // to make the resulting Clift expression valid. In either case, an
        // implicit cast is first applied if necessary.
        if (isByAddressParameter(L)) {
          Argument = emitImplicitCast(Loc, Argument, makePointerType(T));
          Argument = Builder.create<IndirectionOp>(Loc, T, Argument);
        } else {
          Argument = emitImplicitCast(Loc, Argument, T);
        }

        Arguments.push_back(Argument);
      }

      revng_log(ExpressionLog, "CallOp");
      rc_return Builder.create<CallOp>(Loc,
                                       CallType.getReturnType(),
                                       Function,
                                       Arguments);
    }

    if (auto I = llvm::dyn_cast<llvm::SelectInst>(V)) {
      revng_log(ExpressionLog, "llvm::SelectInst");
      LoggerIndent Indent(ExpressionLog);

      mlir::Location Loc = getLocation(I);

      revng_log(ExpressionLog, "Condition subexpression:");
      mlir::Value Condition = (LoggerIndent(ExpressionLog),
                               rc_recur emitExpression(I->getCondition(), Loc));

      revng_log(ExpressionLog, "True subexpression:");
      mlir::Value True = (LoggerIndent(ExpressionLog),
                          rc_recur emitExpression(I->getTrueValue(), Loc));

      revng_log(ExpressionLog, "False subexpression:");
      mlir::Value False = (LoggerIndent(ExpressionLog),
                           rc_recur emitExpression(I->getFalseValue(), Loc));

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

  template<typename OpT, typename... ArgsT>
  OpT emitLocalDeclaration(mlir::Location Loc, ArgsT &&...Args) {
    mlir::OpBuilder::InsertionGuard Guard(Builder);
    Builder.setInsertionPointToEnd(LocalDeclarationBlock);
    OpT Op = Builder.create<OpT>(Loc, std::forward<ArgsT>(Args)...);
    return Op;
  }

  void emitAssignLabel(mlir::Value Label, mlir::Location Loc) {
    Builder.create<AssignLabelOp>(Loc, Label);
  }

  // Determine if an edge from Pred to Succ exists in the scope graph of the
  // current function.
  static bool isScopeGraphEdge(const llvm::BasicBlock *Pred,
                               const llvm::BasicBlock *Succ) {
    for (auto *BB : llvm::children<Scope<const llvm::BasicBlock *>>(Pred)) {
      if (BB == Succ)
        return true;
    }
    return false;
  }

  bool hasBeenEmitted(const llvm::BasicBlock *BB) {
    auto It = BlockMapping.find(BB);
    return It != BlockMapping.end() and It->second.InsertPoint.isSet();
  }

  // Emit a goto targeting the specified basic block.
  // * If the basic block has already been emitted, a label assignment is
  //   inserted at the position matching the beginning of the basic block.
  // * Otherwise, the label declaration is recorded in the basic block table
  //   and the matching assignment will be emitted when the basic block is
  //   emitted.
  void emitGoto(mlir::Location Loc, const llvm::BasicBlock *BB) {
    auto [Iterator, Inserted] = BlockMapping.try_emplace(BB);

    if (not Iterator->second.Label) {
      Iterator->second
        .Label = emitLocalDeclaration<MakeLabelOp>(getLocation(BB));
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
    revng_log(BasicBlockLog, "BB: '" << BB->getName() << "'");
    LoggerIndent Indent(BasicBlockLog);

    // Map BB to the MLIR block, emit label if necessary:
    {
      auto [Iterator, Inserted] = BlockMapping.try_emplace(BB);

      // Save the current insertion point in case a label assignment must later
      // be emitted at the start of this block.
      revng_assert(not Iterator->second.InsertPoint.isSet());
      Iterator->second.InsertPoint = saveInsertionPointAfter(Builder);

      // Sometimes a goto (along a label declaration) targeting this basic block
      // is emitted before the basic block itself. In these cases the label
      // assignment could not be emitted along with the declaration and must be
      // emitted here along with the basic block.
      if (not Inserted and Iterator->second.Label) {
        emitAssignLabel(Iterator->second.Label, getLocation(BB));
        Iterator->second.HasAssignLabel = true;
      }
    }

    const llvm::Instruction *Terminal = BB->getTerminator();
    bool HasGotoMarker = false;

    // Process each instruction contained in this basic block, until the
    // terminal instruction is encountered.
    for (const llvm::Instruction &I : *BB) {
      if (&I == Terminal)
        break;

      // Alloca instructions get special handling and are emitted as local
      // variables in Clift:
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

      // Call instructions require special handling:
      // 1. Some call instructions are only used to represent information about
      //    the scope-graph structure. Such calls are not emitted in Clift.
      // 2. Some calls must only be evaluated once and as such are emitted as
      //    local variable initializers, with any further use of that call value
      //    during expression import being redirected to instead use the
      //    local variable created here.
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

          Iterator->second.TakeAddressOnUse = Indirection;
          continue;
        }
      }

      // Finally, any instruction with no uses represents the root of an
      // expression tree. Any such tree is emitted as an expression statement.
      if (I.use_empty()) {
        mlir::Location Loc = getLocation(&I);
        auto Op = Builder.create<ExpressionStatementOp>(Loc);
        emitExpressionTree(Op.getExpression(), &I, Loc);
      }
    }

    // After the main basic block body is processed, the terminal instruction is
    // handled:

    // Scope-graph goto-markers should only be present in basic blocks
    // terminating with an unconditional branch instruction.
    revng_assert(llvm::isa<llvm::BranchInst>(Terminal) or not HasGotoMarker);
    mlir::Location TerminalLoc = getLocation(Terminal);

    if (llvm::isa<llvm::UnreachableInst>(Terminal)) {
      // Do nothing.
    } else if (auto *Return = llvm::dyn_cast<llvm::ReturnInst>(Terminal)) {
      auto Op = Builder.create<ReturnOp>(TerminalLoc);

      if (const llvm::Value *Value = Return->getReturnValue()) {
        clift::FunctionType FunctionType = CurrentFunction
                                             .getCliftFunctionType();

        clift::ValueType FuncReturnType = FunctionType.getReturnType();
        clift::ValueType LLVMReturnType = FuncReturnType;

        // In SPTAR functions, values are returned by address. In this case
        if (CurrentLayout.hasSPTAR())
          LLVMReturnType = makePointerType(LLVMReturnType);

        // Emit the expression tree rooted at the return instruction directly
        // into the expression region of the newly created return operation:
        emitExpressionTreeImpl(Op.getResult(), [&]() {
          mlir::Value ReturnValue = emitExpression(Value, TerminalLoc);

          // In Clift, arrays cannot be returned, so in the case that the
          // returned expression has array type (knowing that an array value
          // must be an lvalue), we can take its address, and read it as the
          // appropriate type (type punning, violates strict aliasing).
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

          // Emit an implicit cast to the required return type if necessary:
          ReturnValue = emitImplicitCast(TerminalLoc,
                                         ReturnValue,
                                         LLVMReturnType);

          // In the case of SPTAR, because in the LLVM IR the return is by
          // address, but in Clift the return is by value as usual, a final
          // indirection is needed to convert the LLVM IR pointer to a value:
          if (CurrentLayout.hasSPTAR()) {
            ReturnValue = Builder.create<IndirectionOp>(TerminalLoc,
                                                        FuncReturnType,
                                                        ReturnValue);
          }

          return ReturnValue;
        });
      }
    } else if (auto *Branch = llvm::dyn_cast<llvm::BranchInst>(Terminal)) {
      if (Branch->isUnconditional()) {
        const llvm::BasicBlock *Succ = Terminal->getSuccessor(0);

        // In the presence of a scope-graph goto-marker, the unconditional
        // branch is emitted in Clift as a goto. Otherwise it can be considered
        // a continuation of the current scope and emitted recursively.
        if (HasGotoMarker) {
          revng_log(BasicBlockLog,
                    "Explicit goto: '" << Succ->getName() << "'");
          LoggerIndent Indent(BasicBlockLog);
          emitGoto(getLocation(Terminal), Succ);
        } else if (Succ != InnerPostDom) {
          revng_log(BasicBlockLog, "Via unconditional successor...");
          rc_recur emitScope(Succ, InnerPostDom);
        }
      } else {
        // Conditional branches may not be present in a block containing a
        // scope-graph goto-marker. They are emitted as if statements in Clift.
        revng_assert(not HasGotoMarker);
        auto Op = Builder.create<IfOp>(TerminalLoc);

        emitExpressionTree(Op.getCondition(),
                           Branch->getCondition(),
                           TerminalLoc);

        mlir::OpBuilder::InsertionGuard Guard(Builder);

        // Emit true branch:
        revng_assert(Op.getThen().empty());
        Builder.setInsertionPointToEnd(&Op.getThen().emplaceBlock());
        revng_log(BasicBlockLog, "Via BranchInst[true]:");
        (LoggerIndent(BasicBlockLog),
         rc_recur emitScope(Branch->getSuccessor(0), InnerPostDom));

        // Emit false branch:
        revng_assert(Op.getElse().empty());
        Builder.setInsertionPointToEnd(&Op.getElse().emplaceBlock());
        revng_log(BasicBlockLog, "Via BranchInst[false]:");
        (LoggerIndent(BasicBlockLog),
         rc_recur emitScope(Branch->getSuccessor(1), InnerPostDom));
      }
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

        LoggerIndent Indent(BasicBlockLog);
        if (hasBeenEmitted(Succ)) {
          revng_log(BasicBlockLog,
                    "Implicit goto: '" << Succ->getName() << "'");
          emitGoto(TerminalLoc, Succ);
        } else {
          revng_log(BasicBlockLog, "Via SwitchInst[" << CaseValues[I] << "]:");
          rc_recur emitScope(Succ, InnerPostDom);
        }
      }

      // Emit default block:
      if (const llvm::BasicBlock *Succ = Switch->getDefaultDest();
          Succ != nullptr and isScopeGraphEdge(BB, Succ)) {

        revng_assert(Op.getDefaultCaseRegion().empty());
        Builder
          .setInsertionPointToEnd(&Op.getDefaultCaseRegion().emplaceBlock());

        LoggerIndent Indent(BasicBlockLog);
        if (hasBeenEmitted(Succ)) {
          revng_log(BasicBlockLog,
                    "Implicit goto: '" << Succ->getName() << "'");
          emitGoto(TerminalLoc, Succ);
        } else {
          revng_log(BasicBlockLog, "Via SwitchInst[default]:");
          rc_recur emitScope(Succ, InnerPostDom);
        }
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

  /* LLVM function import */

  static llvm::ArrayRef<abi::FunctionType::Layout::Argument>
  getLayoutArguments(const abi::FunctionType::Layout &Layout) {
    llvm::ArrayRef LayoutArguments = Layout.Arguments;
    if (Layout.hasSPTAR())
      LayoutArguments = LayoutArguments.drop_front();
    return LayoutArguments;
  }

public:
  clift::FunctionOp import(const llvm::Function *F) override {
    const model::Function *MF = llvmToModelFunction(Model, *F);
    revng_assert(MF != nullptr);

    revng_log(BasicBlockLog, "Function: '" << F->getName() << "'");
    LoggerIndent Indent(BasicBlockLog);

    auto Op = emitIsolatedFunctionDeclaration(F, MF);
    revng_assert(Op.getBody().empty());
    mlir::Block &BodyBlock = Op.getBody().emplaceBlock();

    CurrentFunction = Op;
    CurrentLayout = abi::FunctionType::Layout::make(*getPrototype(*MF));

    // Clear the function-specific mappings once this function is emitted.
    auto MappingGuard = llvm::make_scope_exit([&]() {
      ArgumentMapping.clear();
      BlockMapping.clear();
      AllocaMapping.clear();
      ValueMapping.clear();
    });

    llvm::ArrayRef LayoutArguments = getLayoutArguments(CurrentLayout);
    revng_assert(F->arg_size() == Op.getArgumentTypes().size());
    revng_assert(F->arg_size() == LayoutArguments.size());

    // Initialize the argument mapping for the function arguments. By-address
    // parameters are detected using the function layout and require indirection
    // before use. In Clift each argument has the type specified in the model.
    // To make the uses of it as imported from the LLVM IR valid, on each use it
    // must be converted to the type imported directly from the LLVM IR argument
    // type. That type is imported here and stored in the argument description.
    for (const auto [A, T, L] :
         llvm::zip(F->args(), Op.getArgumentTypes(), LayoutArguments)) {
      ArgumentMapping.try_emplace(&A,
                                  BodyBlock.addArgument(T, Op->getLoc()),
                                  /*TakeAddressOnUse=*/isByAddressParameter(L),
                                  /*CastType=*/importLLVMType(A.getType()));
    }

    // Compute the scope graph. While the computation should not modify the
    // function IR, it is not const-correct and so a const_cast must be used.
    PostDomTree.recalculate(const_cast<llvm::Function &>(*F));

    mlir::OpBuilder::InsertionGuard BuilderGuard(Builder);
    Builder.setInsertionPointToEnd(&BodyBlock);

    mlir::Block DeclarationBlock;
    ScopedExchange _(LocalDeclarationBlock, &DeclarationBlock);

    // Finally emit the function body starting at the entry block.
    emitScope(&F->getEntryBlock(), PostDomTree.getRootNode()->getBlock());

    BodyBlock.getOperations().splice(BodyBlock.getOperations().begin(),
                                     DeclarationBlock.getOperations());

    return Op;
  }

private:
  mlir::MLIRContext *const Context;
  mlir::ModuleOp CurrentModule;

  const model::Binary &Model;
  mlir::OpBuilder Builder;

  clift::FunctionOp CurrentFunction;
  abi::FunctionType::Layout CurrentLayout;

  mlir::Block *LocalDeclarationBlock = nullptr;

  ScopeGraphPostDomTree PostDomTree;

  clift::ValueType VoidTypeCache;

  uint64_t PointerSize;
  clift::ValueType VoidPointerTypeCache;
  clift::ValueType IntptrTypeCache;

  // While importing the LLVM IR to Clift, we import types from both the model
  // and from the LLVM IR. This creates some discrepancies, particularly around
  // function parameters and return types. Within function bodies we are in the
  // land of LLVM IR types, while any external input (arguments to the current
  // function or return values from called functions) or output (the return
  // value of the current function or arguments to called functions) have types
  // imported from the model. For this reason some conversions must be inserted
  // whenever we cross between the two typing domains.

  struct ArgumentMappingInfo {
    mlir::Value Argument;

    // True if the address of the value should be taken before use in Clift.
    // When converting the model type of an argument to the LLVM IR typing,
    // there may arise a discrepancy in representation: the model parameter may
    // have type large_struct_t, while in the LLVM IR the parameter is passed by
    // address. Taking the address of the argument before each use in the
    // resulting Clift solves this discrepancy.
    bool TakeAddressOnUse;

    // While taking the address solves a discrepancy in representation, a
    // discrepancy in kind may still remain: we might now have large_struct_t*,
    // while the LLVM IR has an integer type. The representation is the same,
    // but the import of a subsequent integer arithmetic operation on the value
    // might fail. For this reason argument values are converted to the type
    // imported directly from LLVM IR, thus ensuring that any subsequently
    // imported expression remains valid.
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

    // See ArgumentMappingInfo::TakeAddressOnUse for more information. However,
    // instead of arguments, this one applies to return values stored in local
    // variables. For some function calls the importer creates local variables
    // with the model typing, and when the LLVM IR function call returned by
    // address, the address of the variable must be taken to resolve the
    // discrepancy in representation.
    bool TakeAddressOnUse;

    ValueMappingInfo(mlir::Value Value) :
      Value(Value), TakeAddressOnUse(false) {}
  };

  /* Per-module mappings - persisted between imported functions */

  // Maps LLVM object to a Clift global op. Each used global object is only
  // emitted once and hence forth the operation stored in this table is used.
  llvm::DenseMap<const llvm::GlobalObject *, clift::GlobalOpInterface>
    SymbolMapping;

  // Maps LLVM object to a string literal description. Used for caching
  // successful string literal detection results.
  llvm::DenseMap<const llvm::GlobalVariable *, StringLiteral> StringMapping;

  // Maps helper name to Clift function op. Each helper function declaration is
  // only emitted once.
  llvm::DenseMap<llvm::StringRef, clift::FunctionOp> HelperMapping;

  /* Per-function mappings - cleared for each imported function */

  // Maps LLVM IR function argument to a Clift argument description. Initialized
  // before importing each function body. During expression import argument
  // values are generated using the descriptions stored in this table.
  llvm::DenseMap<const llvm::Argument *, ArgumentMappingInfo> ArgumentMapping;

  // Maps LLVM IR basic block to a Clift import state for that block.
  llvm::DenseMap<const llvm::BasicBlock *, BlockMappingInfo> BlockMapping;

  // Maps LLVM IR alloca to a Clift local variable op result. Used during
  // expression import to map uses of an alloca to the matching Clift local
  // variable.
  llvm::DenseMap<const llvm::AllocaInst *, mlir::Value> AllocaMapping;

  // Maps an arbitrary LLVM IR value to a Clift value description. During
  // expression import some values are only emitted once and any use of such a
  // value is emitted using the description stored in this table.
  llvm::DenseMap<const llvm::Value *, ValueMappingInfo> ValueMapping;
};

} // namespace

std::unique_ptr<Clifter> Clifter::make(mlir::ModuleOp Module,
                                       const model::Binary &Model) {
  return std::make_unique<ClifterImpl>(Module, Model);
}
