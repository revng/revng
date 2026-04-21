//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/GenericDomTree.h"

#include "revng/ABI/FunctionType/Layout.h"
#include "revng/ADT/RecursiveCoroutine.h"
#include "revng/ADT/ScopedExchange.h"
#include "revng/Clift/CliftAttributes.h"
#include "revng/Clift/CliftDialect.h"
#include "revng/Clift/CliftTypes.h"
#include "revng/CliftImportModel/ImportModel.h"
#include "revng/Clifter/Clifter.h"
#include "revng/LocalVariables/LocalVariableHelpers.h"
#include "revng/Model/Architecture.h"
#include "revng/Model/Binary.h"
#include "revng/Model/FunctionTags.h"
#include "revng/Model/IRHelpers.h"
#include "revng/Model/NameBuilder.h"
#include "revng/Pipeline/Location.h"
#include "revng/Pipes/Ranks.h"
#include "revng/RestructureCFG/ScopeGraphGraphTraits.h"
#include "revng/Support/Debug.h"
#include "revng/Support/IRHelpers.h"
#include "revng/Support/Identifier.h"

using namespace clift;

namespace {

static Logger BasicBlockLog{ "clifter-basic-blocks" };
static Logger ExpressionLog{ "clifter-expressions" };

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
  class FunctionClifter;

  struct StringLiteral {
    clift::ArrayType Type;
    std::string Data;
  };

  mlir::MLIRContext *const Context;
  mlir::ModuleOp CurrentModule;

  const model::Binary &Model;
  mlir::OpBuilder Builder;

  const llvm::DataLayout *DataLayout;

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

public:
  // It is important only to query minimal properties about the model in the
  // importer constructor to avoid all imported functions depending on those
  // properties.
  explicit ClifterImpl(mlir::ModuleOp Module, const model::Binary &Model) :
    Context(Module.getContext()),
    CurrentModule(Module),
    Model(Model),
    Builder(Context),
    DataLayout(nullptr) {
    revng_assert(clift::isCliftModule(Module));
    Builder.setInsertionPointToEnd(Module.getBody());
  }

private:
  //===---------------------------- Debug info ----------------------------===//

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

  //===-------------------------- Type utilities --------------------------===//

  static uint64_t getIntegerSize(unsigned IntegerWidth) {
    if (IntegerWidth == 1)
      IntegerWidth = 8;

    revng_check(IntegerWidth % 8 == 0);

    // Convert the size in bits to a size in octets:
    return IntegerWidth / 8;
  }

  IntegerType getIntegerType(uint64_t Size,
                             IntegerKind Kind = IntegerKind::Generic) const {
    return IntegerType::get(Context, Kind, Size);
  }

  uint64_t getModelPointerSize() const {
    return model::Architecture::getPointerSize(Model.Architecture());
  }

  mlir::Type getModelPointerType(mlir::Type ElementType) const {
    return PointerType::get(ElementType, getModelPointerSize());
  }

  uint64_t getPointerSize() const {
    revng_assert(DataLayout->getPointerSizeInBits() % 8 == 0);
    return DataLayout->getPointerSize();
  }

  mlir::Type getPointerType(mlir::Type ElementType) const {
    return PointerType::get(ElementType, getPointerSize());
  }

  mlir::Type getVoidType() const { return VoidType::get(Context); }

  mlir::Type getVoidPointerType() const {
    return getPointerType(getVoidType());
  }

  mlir::Type getIntptrType() { return getIntegerType(getPointerSize()); }

  clift::ValueType getConstCharType() {
    return clift::IntegerType::get(Context,
                                   IntegerKind::Number,
                                   1,
                                   /*IsConst=*/true);
  }

  clift::ArrayType getConstCharArrayType(uint64_t NumElements) {
    return clift::ArrayType::get(getConstCharType(), NumElements);
  }

  clift::ValueType getCharType() {
    return clift::IntegerType::get(Context,
                                   IntegerKind::Number,
                                   1,
                                   /*IsConst=*/false);
  }

  clift::ArrayType getCharArrayType(uint64_t NumElements) {
    return clift::ArrayType::get(getCharType(), NumElements);
  }

  template<typename FunctionT>
  model::UpcastableType getPrototype(const FunctionT &Function) {
    if (auto &&Type = Function.Prototype())
      return Type;
    return Model.makeType(Model.defaultPrototype()->key());
  }

  //===------------------------- Model type import ------------------------===//

  template<typename TypeT, typename ModelTypeT>
  TypeT importType(const ModelTypeT &Type) {
    return mlir::cast<TypeT>(clift::importType(Context, Type));
  }

  //===------------------------- LLVM type import -------------------------===//

  IntegerType importLLVMIntegerType(const llvm::IntegerType *Type,
                                    IntegerKind Kind = IntegerKind::Generic) {
    return getIntegerType(getIntegerSize(Type->getBitWidth()), Kind);
  }

  mlir::Type importLLVMPointerType(const llvm::PointerType *Type) {
    revng_assert(Type->getAddressSpace() == 0);
    return getVoidPointerType();
  }

  mlir::Type importLLVMType(const llvm::Type *Type) {
    if (Type->isVoidTy())
      return getVoidType();

    if (auto *T = llvm::dyn_cast<llvm::IntegerType>(Type))
      return importLLVMIntegerType(T);

    if (auto *T = llvm::dyn_cast<llvm::PointerType>(Type))
      return importLLVMPointerType(T);

    if (auto *T = llvm::dyn_cast<llvm::ArrayType>(Type)) {
      const auto *ElementType = T->getElementType();
      revng_assert(ElementType->isIntegerTy());
      revng_assert(ElementType->getIntegerBitWidth() == 8);
      uint64_t NumBytes = T->getNumElements()
                          * (ElementType->getIntegerBitWidth() / 8);

      return clift::makeOpaqueStruct(*Context, NumBytes);
    }

    if (auto *S = llvm::dyn_cast<llvm::StructType>(Type)) {
      const auto *StructLayout = DataLayout->getStructLayout(S);
      uint64_t StructByteSize = StructLayout->getSizeInBytes();

      uint64_t PreviousOffsetInBits = 0ULL;

      for (auto [Index, FieldType] : llvm::enumerate(S->elements())) {
        revng_assert(PreviousOffsetInBits % 8 == 0);

        uint64_t OffsetInBits = StructLayout->getElementOffsetInBits(Index);
        revng_assert(OffsetInBits % 8 == 0);

        uint64_t FieldSizeInBits = DataLayout->getTypeSizeInBits(FieldType);
        revng_assert(FieldSizeInBits % 8 == 0,
                     ("StructType with a field whose size is not a multiple of "
                      " 8 bits. Field index: "
                      + std::to_string(Index)
                      + ". FieldSizeInBits: " + std::to_string(FieldSizeInBits))
                       .c_str());

        PreviousOffsetInBits = OffsetInBits + FieldSizeInBits;
      }
      revng_assert(PreviousOffsetInBits % 8 == 0);
      uint64_t Alignment = StructLayout->getAlignment().value();
      uint64_t ByteSize = PreviousOffsetInBits / 8;
      revng_assert(llvm::alignTo(ByteSize, Alignment) == StructByteSize);
      revng_assert(StructByteSize);
      return clift::makeOpaqueStruct(*Context, StructByteSize);
    }

    revng_abort("Unsupported LLVM type");
  }

  //===------------------------ LLVM global import ------------------------===//

  uint64_t getConstantInt(const llvm::Value *Value) {
    return llvm::cast<llvm::ConstantInt>(Value)->getZExtValue();
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
  mlir::Type importHelperReturnType(const llvm::Function *HelperFunction) {

    const llvm::FunctionType *HelperType = HelperFunction->getFunctionType();
    const llvm::Type *ReturnType = HelperType->getReturnType();
    auto Struct = llvm::dyn_cast<llvm::StructType>(ReturnType);

    if (not Struct)
      return importLLVMType(ReturnType);

    revng_assert(Struct->isLiteral());

    llvm::StringRef HelperName = HelperFunction->getName();
    auto Location = pipeline::location(revng::ranks::HelperStructType,
                                       HelperName.str());

    llvm::SmallVector<FieldAttr> Fields;
    Fields.reserve(Struct->getNumElements());

    // For this artificial struct we have to use the byte size and the field
    // offsets provided by the LLVM DataLayout, because that's how LLVM reasons
    // about this struct anyway, and that's how all the GEPs that index into an
    // alloca with this StructType will reason about. Using a size and offsets
    // will break semantics, because the semantic used by all of LLVM's analysis
    // and transformations is the semantic dictated by the DataLayout.
    const auto &DataLayout = HelperFunction->getParent()->getDataLayout();
    const auto *StructLayout = DataLayout.getStructLayout(Struct);
    uint64_t StructByteSize = StructLayout->getSizeInBytes();

    uint64_t PreviousOffsetInBits = 0ULL;
    for (auto [Index, T] : llvm::enumerate(Struct->elements())) {
      mlir::Type FieldType = importLLVMType(T);
      revng_assert(PreviousOffsetInBits % 8 == 0);

      uint64_t FieldOffsetInBits = StructLayout->getElementOffsetInBits(Index);
      revng_assert(FieldOffsetInBits % 8 == 0);

      std::string FieldName = getHelperStructFieldName(Index);
      std::string FieldHandle = Location
                                  .extend(revng::ranks::HelperStructField,
                                          FieldName)
                                  .toString();

      Fields.push_back(FieldAttr::get(Context,
                                      FieldHandle,
                                      makeNameAttr<FieldAttr>(Context,
                                                              FieldHandle,
                                                              FieldName),
                                      makeCommentAttr<FieldAttr>(Context,
                                                                 FieldHandle),
                                      FieldOffsetInBits / 8,
                                      FieldType));

      uint64_t FieldSizeInBits = DataLayout.getTypeSizeInBits(T);
      revng_assert(FieldSizeInBits % 8 == 0,
                   (HelperFunction->getName().str()
                    + " returns a StructType with a field whose size "
                      "is not a multiple of 8 bits. Field index: "
                    + std::to_string(Index)
                    + ". FieldSizeInBits: " + std::to_string(FieldSizeInBits))
                     .c_str());

      PreviousOffsetInBits = FieldOffsetInBits + FieldSizeInBits;
    }
    revng_assert(PreviousOffsetInBits % 8 == 0);
    uint64_t Alignment = StructLayout->getAlignment().value();
    uint64_t ByteSize = PreviousOffsetInBits / 8;
    revng_assert(llvm::alignTo(ByteSize, Alignment) == StructByteSize);
    revng_assert(StructByteSize);

    auto Handle = pipeline::locationString(revng::ranks::HelperStructType,
                                           HelperName.str());

    std::string
      Name = Model.Configuration().Naming().ArtificialReturnValuePrefix()
             + sanitizeIdentifier(HelperName);

    auto Definition = StructAttr::get(Context,
                                      Handle,
                                      makeNameAttr<StructAttr>(Context,
                                                               Handle,
                                                               Name),
                                      makeCommentAttr<StructAttr>(Context,
                                                                  Handle),
                                      StructByteSize,
                                      Fields,
                                      {});

    return StructType::get(Context, Definition);
  }

  // Import a Clift function type from an LLVM function, using the name of the
  // function for deriving the type handle.
  clift::FunctionType importHelperType(const llvm::Function *HelperFunction) {
    const llvm::FunctionType *HelperType = HelperFunction->getFunctionType();
    revng_assert(not HelperType->isVarArg());

    mlir::Type ReturnType = importHelperReturnType(HelperFunction);

    llvm::SmallVector<mlir::Type> ParameterTypes;
    ParameterTypes.reserve(HelperType->getNumParams());

    for (const llvm::Type *T : HelperType->params())
      ParameterTypes.push_back(importLLVMType(T));

    auto Handle = pipeline::locationString(revng::ranks::HelperFunction,
                                           HelperFunction->getName().str());

    return FunctionType::get(Context,
                             Handle,
                             makeNameAttr<FunctionType>(Context, Handle),
                             makeCommentAttr<FunctionType>(Context, Handle),
                             ReturnType,
                             ParameterTypes,
                             {});
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
      revng_assert(Iterator->second.getFunctionType() == FunctionType);
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

  FunctionOp getHelperFunction(llvm::StringRef HelperName,
                               clift::FunctionType FunctionType) {
    return emitHelperFunctionDeclaration(mlir::UnknownLoc::get(Context),
                                         HelperName,
                                         FunctionType);
  }

  template<typename FunctionT, typename RankT, typename... ArgsT>
  clift::FunctionOp emitModelFunctionDeclaration(const llvm::Function *F,
                                                 const FunctionT &MF,
                                                 const RankT &Rank,
                                                 ArgsT &&...Args) {
    model::UpcastableType Prototype = getPrototype(MF);
    auto FunctionType = importType<clift::FunctionType>(*Prototype);
    auto Handle = pipeline::locationString(Rank, std::forward<ArgsT>(Args)...);

    // NOTE: `unwrap()` explicitly marks the entire container to have been read
    //        exactly (so *any* change to it triggers invalidation).
    auto Attributes = MF.Attributes().unwrap();

    // NOTE: no matter what, do not capture `MF`!
    auto ImportImpl = [F, FunctionType, Handle, Attributes, this]() // format
      -> clift::FunctionOp {
      // WARNING: DO NOT QUERY MODEL PROPERTIES WITHIN THIS SCOPE.
      //
      // It is very important not to query any tracked model properties within
      // this scope, as doing so would break invalidation when the import of
      // a function uses a declaration already emitted during the import of
      // a previous function.
      //
      // To be more precise, this problem arises when:
      //
      // 1. The Clifter imports the definition of function F1, where F1 uses
      //    some other function G. At this point a declaration of G is created
      //    in this scope.
      //
      // 2. The Clifter imports the definition of function F2, where F2 also
      //    uses G. At this point the declaration of G has already been created
      //    and this scope is not entered a second time.
      //
      // If model properties were queried within this scope, those accesses
      // would not be tracked during the import of the definition of F2, and
      // so changes to G would fail to correctly invalidate F2.
      return importFunctionDeclaration(CurrentModule,
                                       getLocation(F->getSubprogram()),
                                       F->getName(),
                                       Handle,
                                       FunctionType,
                                       Attributes);
    };
    return getOrEmitSymbol(F, ImportImpl);
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
    bool HasPrefix = Name.consume_front("dynamic_");
    revng_check(HasPrefix);

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

  mlir::Type emitGlobalObject(const llvm::GlobalObject *G) {
    if (const auto *F = llvm::dyn_cast<llvm::Function>(G))
      return emitFunctionDeclaration(F).getFunctionType();

    revng_abort("Unsupported global object kind");
  }

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

    String.Type = getConstCharArrayType(Type->getNumElements());

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

public:
  clift::FunctionOp import(const llvm::Function *F) override;
};

class ClifterImpl::FunctionClifter {
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

  ClifterImpl &C;

  mlir::MLIRContext *const Context;
  mlir::OpBuilder Builder;

  clift::FunctionOp Function;
  const model::Function &ModelFunction;
  abi::FunctionType::Layout FunctionLayout;
  ScopeGraphPostDomTree PostDomTree;

  uint64_t NextLocalVariableIndex = 0;
  uint64_t NextGotoLabelIndex = 0;

  mlir::Block LocalDeclarationBlock;

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

  // Maps an LLVM CallInst that returns SPTAR to the Clift LocalVariableOp that
  // holds the return value.
  // All LLVM uses of the CallInst use it as a pointer, as if it was pointing to
  // the storage holding the return value. This is a hack, and should be avoided
  // in the future. Until then, for the generation of uses of the SPTAR call in
  // Clift, the address of the LocalVariablOp must be taken, to preserve
  // semantic.
  llvm::DenseMap<const llvm::CallInst *, mlir::Value> SPTARCallVariableMapping;

public:
  static clift::FunctionOp import(ClifterImpl &C, const llvm::Function *F) {
    const model::Function *MF = llvmToModelFunction(C.Model, *F);
    revng_assert(MF != nullptr);

    revng_log(BasicBlockLog, "Function: '" << F->getName() << "'");
    LoggerIndent Indent(BasicBlockLog);

    auto NullReset = ScopedExchange(C.DataLayout,
                                    &F->getParent()->getDataLayout());
    auto Function = C.emitIsolatedFunctionDeclaration(F, MF);
    return FunctionClifter(C, Function, *MF).import(F);
  }

private:
  explicit FunctionClifter(ClifterImpl &Clifter,
                           clift::FunctionOp Function,
                           const model::Function &ModelFunction) :
    C(Clifter),
    Context(Clifter.Context),
    Builder(Clifter.Context),
    Function(Function),
    ModelFunction(ModelFunction),
    FunctionLayout(makeLayout(Clifter, ModelFunction)) {}

  static abi::FunctionType::Layout makeLayout(ClifterImpl &Clifter,
                                              const model::Function &Model) {
    return abi::FunctionType::Layout::make(*Clifter.getPrototype(Model));
  }

  //===---------------------- LLVM expression import ----------------------===//

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

  template<typename OpT, typename... ArgsT>
  mlir::Value emitExpr(mlir::Location Loc, ArgsT &&...Args) {
    return Builder.create<OpT>(Loc, std::forward<ArgsT>(Args)...);
  }

  template<typename OpT>
  mlir::Value
  emitCast(mlir::Location Loc, mlir::Value Value, mlir::Type TargetType) {
    if (Value.getType() != TargetType) {
      OpT X = Builder.create<OpT>(Loc, TargetType, Value);
      if constexpr (std::is_same_v<OpT, BitCastOp>) {
        auto ValueSize = unwrapped_cast<ValueType>(Value.getType())
                           .getObjectSize();
        auto TargetSize = unwrapped_cast<ValueType>(TargetType).getObjectSize();
        revng_assert(ValueSize == TargetSize);
      }
      return X;
    }

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
  mlir::Value emitImplicitBitcast(mlir::Location Loc,
                                  mlir::Value Value,
                                  mlir::Type TargetType) {
    mlir::Type SourceType = Value.getType();

    if (SourceType == TargetType)
      return Value;

    auto SourceT = clift::unwrapped_dyn_cast<ValueType>(SourceType);
    if (not SourceT)
      return Value;

    return emitCast<BitCastOp>(Loc, Value, TargetType);
  }

  // Used for emitting operations acting on integer operands (arithmetic,
  // comparison, etc.) of a specific kind. Before applying the operation, the
  // operands are automatically converted to the requested kind, if necessary.
  // After the operation, the result is automaticlaly converted to generic kind.
  mlir::Value emitIntegerOp(mlir::Location Loc,
                            IntegerKind Kind,
                            auto ApplyOperation,
                            std::same_as<mlir::Value> auto... Operands) {
    auto ConvertToKind = [&](mlir::Value &Value, IntegerKind Kind) {
      auto UnderlyingType = getUnderlyingIntegerType(Value.getType());
      revng_assert(UnderlyingType);

      uint64_t Size = UnderlyingType.getSize();
      Value = emitCast<BitCastOp>(Loc, Value, C.getIntegerType(Size, Kind));
    };

    (ConvertToKind(Operands, Kind), ...);
    mlir::Value Result = ApplyOperation(Operands...);

    if (Kind != IntegerKind::Generic)
      ConvertToKind(Result, IntegerKind::Generic);

    return Result;
  }

  // Applies an integer cast of the operand to the specified size, using the
  // specified primitive kind. Finally the result is converted to generic kind.
  mlir::Value emitIntegerCast(mlir::Location Loc,
                              mlir::Value Operand,
                              uint64_t Size,
                              IntegerKind Kind = IntegerKind::Generic) {
    uint64_t SrcSize = getUnderlyingIntegerType(Operand.getType()).getSize();

    auto EmitCast = [&](mlir::Value Operand) {
      auto NewType = C.getIntegerType(Size, Kind);

      if (Size > SrcSize)
        return emitCast<ExtendOp>(Loc, Operand, NewType);

      if (Size < SrcSize)
        return emitCast<TruncateOp>(Loc, Operand, NewType);

      return emitCast<BitCastOp>(Loc, Operand, NewType);
    };

    return emitIntegerOp(Loc, Kind, EmitCast, Operand);
  }

  mlir::Value useGlobal(mlir::Location Loc, clift::GlobalOpInterface Global) {
    return Builder.create<UseOp>(Loc, Global.getType(), Global.getName());
  }

  mlir::Value emitHelperCall(mlir::Location Loc,
                             FunctionOp Function,
                             llvm::ArrayRef<mlir::Value> Arguments) {
    auto FunctionType = Function.getFunctionType();

    llvm::SmallVector<mlir::Value> CastArgs;
    CastArgs.reserve(Arguments.size());

    for (auto [A, T] : llvm::zip(Arguments, FunctionType.getArgumentTypes()))
      CastArgs.push_back(emitImplicitBitcast(Loc, A, T));

    return Builder.create<CallOp>(Loc,
                                  FunctionType.getReturnType(),
                                  useGlobal(Loc, Function),
                                  CastArgs);
  }

  mlir::Value emitHelperCall(mlir::Location Loc,
                             const llvm::Function *F,
                             llvm::ArrayRef<mlir::Value> Arguments) {
    return emitHelperCall(Loc, C.emitHelperFunctionDeclaration(F), Arguments);
  }

  mlir::Value emitHelperCall(mlir::Location Loc,
                             llvm::StringRef HelperName,
                             mlir::Type ReturnType,
                             llvm::ArrayRef<mlir::Type> ParameterTypes,
                             llvm::ArrayRef<mlir::Value> Arguments) {
    auto Handle = pipeline::locationString(revng::ranks::HelperFunction,
                                           HelperName.str());

    auto NameAttr = makeNameAttr<clift::FunctionType>(Context, Handle);
    auto CommentAttr = makeCommentAttr<clift::FunctionType>(Context, Handle);

    // TODO: should we add something explicitly identifying this as a helper?
    llvm::ArrayRef<clift::CAttributeAttr> Attributes = {};

    auto FunctionType = clift::FunctionType::get(Context,
                                                 Handle,
                                                 NameAttr,
                                                 CommentAttr,
                                                 ReturnType,
                                                 ParameterTypes,
                                                 Attributes);

    return emitHelperCall(Loc,
                          C.getHelperFunction(HelperName, FunctionType),
                          Arguments);
  }

  RecursiveCoroutine<mlir::Value>
  emitStructInitializer(const llvm::CallInst *Call) {
    mlir::Location Loc = C.getLocation(Call);

    auto UseBegin = Call->use_begin();
    auto UseEnd = Call->use_end();
    revng_assert(UseBegin != UseEnd);

    // StructInitializer must have exactly one user of type ReturnInst.
    revng_assert(llvm::isa<llvm::ReturnInst>(UseBegin->getUser()));
    revng_assert(std::next(UseBegin) == UseEnd);

    auto T = mlir::dyn_cast<StructType>(Function.getReturnType());
    revng_assert(Call->arg_size() == T.getFields().size());

    llvm::SmallVector<mlir::Value> Initializers;
    for (const auto &[A, F] : llvm::zip(Call->args(), T.getFields())) {
      mlir::Value Value = rc_recur emitExpression(A, Loc);
      Value = emitImplicitBitcast(Loc, Value, F.getType());
      Initializers.push_back(Value);
    }

    rc_return Builder.create<AggregateOp>(Loc, T, Initializers);
  }

  RecursiveCoroutine<mlir::Value> emitHelperCall(const llvm::CallInst *Call) {
    // TODO: Use getCalledFunction. Pending fix for uniqued-by-prototype issue.
    // const llvm::Function *Callee = Call->getCalledFunction();
    auto *Callee = llvm::dyn_cast<llvm::Function>(Call->getCalledOperand());
    revng_assert(Callee != nullptr);

    auto Tags = FunctionTags::TagsSet::from(Callee);

    if (Tags.contains(FunctionTags::StructInitializer))
      rc_return rc_recur emitStructInitializer(Call);

    mlir::Location Loc = C.getLocation(Call);

    llvm::SmallVector<mlir::Value> Arguments;
    Arguments.reserve(Call->arg_size());

    for (const llvm::Value *Argument : Call->args())
      Arguments.push_back(rc_recur emitExpression(Argument, Loc));

    rc_return emitHelperCall(Loc,
                             C.emitHelperFunctionDeclaration(Callee),
                             Arguments);
  }

  RecursiveCoroutine<mlir::Value>
  emitExpression(const llvm::Value *V, mlir::Location SurroundingLocation) {
    LoggerIndent Indent(ExpressionLog);

    if (auto G = llvm::dyn_cast<llvm::GlobalObject>(V)) {
      if (const StringLiteral *String = C.detectStringLiteral(G)) {
        revng_log(ExpressionLog, "llvm::GlobalObject -> StringOp");
        auto Op = Builder.create<StringOp>(SurroundingLocation,
                                           String->Type,
                                           std::move(String->Data));

        auto Type = C.getPointerType(String->Type.getElementType());
        rc_return emitCast<DecayOp>(SurroundingLocation, Op, Type);
      }

      if (const auto *F = llvm::dyn_cast<llvm::Function>(G)) {
        revng_log(ExpressionLog, "llvm::Function -> UseOp");
        auto Function = C.emitFunctionDeclaration(F);
        rc_return Builder.create<UseOp>(SurroundingLocation,
                                        Function.getFunctionType(),
                                        Function.getSymNameAttr());
      }

      revng_abort("Unsupported global object kind");
    }

    if (auto A = llvm::dyn_cast<llvm::Argument>(V)) {
      revng_log(ExpressionLog, "llvm::Argument");
      LoggerIndent Indent(ExpressionLog);

      auto It = ArgumentMapping.find(A);
      revng_assert(It != ArgumentMapping.end());

      mlir::Value Value = It->second.Argument;
      bool TakeAddressOnUse = It->second.TakeAddressOnUse;
      mlir::Type ArgumentUseType = It->second.CastType;

      if (TakeAddressOnUse) {
        uint64_t UseSize = unwrapped_cast<IntegerType>(ArgumentUseType)
                             .getObjectSize();
        revng_assert(C.getModelPointerSize() == UseSize);
        Value = Builder
                  .create<AddressofOp>(SurroundingLocation,
                                       C.getModelPointerType(Value.getType()),
                                       Value);
      }

      rc_return emitImplicitBitcast(SurroundingLocation,
                                    Value,
                                    It->second.CastType);
    }

    if (auto U = llvm::dyn_cast<llvm::UndefValue>(V)) {
      revng_log(ExpressionLog, "llvm::UndefValue -> UndefOp");
      mlir::Type Type = C.importLLVMType(U->getType());
      rc_return Builder.create<UndefOp>(SurroundingLocation, Type);
    }

    if (auto I = llvm::dyn_cast<llvm::ConstantInt>(V)) {
      revng_log(ExpressionLog, "llvm::ConstantInt -> ImmediateOp");
      const llvm::IntegerType *T = llvm::cast<llvm::IntegerType>(I->getType());
      rc_return Builder.create<ImmediateOp>(SurroundingLocation,
                                            C.importLLVMIntegerType(T),
                                            I->getZExtValue());
    }

    if (auto N = llvm::dyn_cast<llvm::ConstantPointerNull>(V)) {
      revng_log(ExpressionLog, "llvm::ConstantPointerNull -> ImmediateOp");
      auto Op = Builder.create<ImmediateOp>(SurroundingLocation,
                                            C.getIntptrType(),
                                            /*Value=*/0);

      LoggerIndent Indent(ExpressionLog);
      rc_return emitCast<BitCastOp>(SurroundingLocation,
                                    Op,
                                    C.getVoidPointerType());
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

      revng_log(ExpressionLog, "llvm::AllocaInst");
      LoggerIndent Indent(ExpressionLog);

      auto It = AllocaMapping.find(I);
      revng_assert(It != AllocaMapping.end());

      auto Type = C.getPointerType(It->second.getType());
      auto Value = Builder.create<AddressofOp>(Loc, Type, It->second);
      rc_return emitImplicitBitcast(Loc, Value, C.importLLVMType(I->getType()));
    }

    if (auto I = llvm::dyn_cast<llvm::LoadInst>(V)) {
      revng_log(ExpressionLog, "llvm::LoadInst");
      LoggerIndent Indent(ExpressionLog);

      mlir::Location Loc = C.getLocation(I);

      revng_log(ExpressionLog, "Subexpression:");
      mlir::Value Pointer = (LoggerIndent(ExpressionLog),
                             rc_recur emitExpression(I->getPointerOperand(),
                                                     Loc));

      mlir::Type ValueType = C.importLLVMType(V->getType());
      mlir::Type PointerType = C.getPointerType(ValueType);

      Pointer = Builder.create<BitCastOp>(Loc, PointerType, Pointer);

      revng_log(ExpressionLog, "IndirectionOp");
      rc_return Builder.create<IndirectionOp>(Loc, ValueType, Pointer);
    }

    if (auto I = llvm::dyn_cast<llvm::StoreInst>(V)) {
      revng_log(ExpressionLog, "llvm::StoreInst");
      LoggerIndent Indent(ExpressionLog);

      mlir::Location Loc = C.getLocation(I);

      revng_log(ExpressionLog, "Pointer subexpression:");
      mlir::Value Pointer = (LoggerIndent(ExpressionLog),
                             rc_recur emitExpression(I->getPointerOperand(),
                                                     Loc));

      revng_log(ExpressionLog, "Value subexpression:");
      mlir::Value Value = (LoggerIndent(ExpressionLog),
                           rc_recur emitExpression(I->getValueOperand(), Loc));

      Pointer = Builder.create<BitCastOp>(Loc,
                                          C.getPointerType(Value.getType()),
                                          Pointer);

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

      mlir::Location Loc = C.getLocation(I);

      revng_log(ExpressionLog, "LHS subexpression");
      mlir::Value Lhs = (LoggerIndent(ExpressionLog),
                         rc_recur emitExpression(I->getOperand(0), Loc));

      revng_log(ExpressionLog, "RHS subexpression");
      mlir::Value Rhs = (LoggerIndent(ExpressionLog),
                         rc_recur emitExpression(I->getOperand(1), Loc));

      IntegerKind Kind = IntegerKind::Generic;
      switch (I->getOpcode()) {
      case Operators::SDiv:
      case Operators::SRem:
      case Operators::AShr:
        Kind = IntegerKind::Signed;
        break;
      case Operators::UDiv:
      case Operators::URem:
      case Operators::LShr:
        Kind = IntegerKind::Unsigned;
        break;
      default:
        break;
      }

      auto *IntegerType = llvm::cast<llvm::IntegerType>(V->getType());
      auto Type = C.importLLVMIntegerType(IntegerType, Kind);

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

      IntegerKind Kind = IntegerKind::Generic;
      switch (I->getPredicate()) {
      case ICMP_SGT:
      case ICMP_SGE:
      case ICMP_SLT:
      case ICMP_SLE:
        Kind = IntegerKind::Signed;
        break;
      case ICMP_UGT:
      case ICMP_UGE:
      case ICMP_ULT:
      case ICMP_ULE:
        Kind = IntegerKind::Unsigned;
        break;
      default:
        break;
      }

      mlir::Location Loc = C.getLocation(I);

      revng_log(ExpressionLog, "LHS subexpression");
      mlir::Value Lhs = (LoggerIndent(ExpressionLog),
                         rc_recur emitExpression(I->getOperand(0), Loc));

      revng_log(ExpressionLog, "RHS subexpression");
      mlir::Value Rhs = (LoggerIndent(ExpressionLog),
                         rc_recur emitExpression(I->getOperand(1), Loc));

      auto *IntegerType = llvm::cast<llvm::IntegerType>(V->getType());
      auto Type = C.importLLVMIntegerType(IntegerType, IntegerKind::Signed);

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

      // Pointer comparisons are handled separately. Non-signed comparisons are
      // emitted directly as pointer comparisons. For signed comparisons, the
      // pointer operands are converted to integers before the comparison.
      if (llvm::isa<llvm::PointerType>(I->getOperand(0)->getType())) {
        if (not I->isSigned())
          rc_return EmitOp(Lhs, Rhs);

        Lhs = emitCast<BitCastOp>(Loc, Lhs, C.getIntptrType());
        Rhs = emitCast<BitCastOp>(Loc, Rhs, C.getIntptrType());
      }

      rc_return emitIntegerOp(Loc, Kind, EmitOp, Lhs, Rhs);
    }

    if (auto I = llvm::dyn_cast<llvm::CastInst>(V)) {
      revng_log(ExpressionLog, "llvm::CastInst");
      LoggerIndent Indent(ExpressionLog);

      mlir::Location Loc = C.getLocation(I);

      revng_log(ExpressionLog, "Subexpression:");
      mlir::Value Operand = (LoggerIndent(ExpressionLog),
                             rc_recur emitExpression(I->getOperand(0), Loc));

      auto GetIntegerSize = [](const llvm::Type *Type) {
        const auto *IntegerType = llvm::cast<llvm::IntegerType>(Type);
        return ClifterImpl::getIntegerSize(IntegerType->getBitWidth());
      };

      auto EmitIntegerCast = [&](IntegerKind Kind) {
        return emitIntegerCast(Loc,
                               Operand,
                               GetIntegerSize(V->getType()),
                               Kind);
      };

      switch (I->getOpcode()) {
        using Operators = llvm::CastInst::CastOps;
      case Operators::SExt:
        rc_return EmitIntegerCast(IntegerKind::Signed);
      case Operators::ZExt:
      case Operators::Trunc:
        rc_return EmitIntegerCast(IntegerKind::Generic);
      case Operators::PtrToInt:
        Operand = emitCast<BitCastOp>(Loc, Operand, C.getIntptrType());
        if (uint64_t S = GetIntegerSize(I->getDestTy());
            S != C.getPointerSize())
          Operand = emitIntegerCast(Loc, Operand, S);
        rc_return Operand;
      case Operators::IntToPtr:
        if (uint64_t S = GetIntegerSize(I->getSrcTy()); S != C.getPointerSize())
          Operand = emitIntegerCast(Loc, Operand, C.getPointerSize());
        rc_return emitCast<BitCastOp>(Loc, Operand, C.getVoidPointerType());
      default:
        revng_abort("Unsupported LLVM cast operation.");
      }
    }

    if (auto I = llvm::dyn_cast<llvm::CallInst>(V)) {
      revng_log(ExpressionLog, "llvm::CallInst");
      LoggerIndent Indent(ExpressionLog);

      mlir::Location Loc = C.getLocation(I);

      if (auto It = SPTARCallVariableMapping.find(I);
          It != SPTARCallVariableMapping.end()) {
        revng_log(ExpressionLog, "<existing value>");

        mlir::Value Value = It->second;

        Value = Builder.create<AddressofOp>(SurroundingLocation,
                                            C.getPointerType(Value.getType()),
                                            Value);

        rc_return Value;
      }

      // Any call without a prototype is considered a helper function and is
      // imported separately:
      if (not I->hasMetadata(PrototypeMDName))
        rc_return emitHelperCall(I);

      const auto *ModelCallType = getCallSitePrototype(C.Model, I);
      auto Layout = abi::FunctionType::Layout::make(*ModelCallType);

      auto CallType = C.importType<clift::FunctionType>(*ModelCallType);

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
          Function = emitCast<DecayOp>(Loc,
                                       Function,
                                       C.getPointerType(FunctionType));
        }

        Function = emitCast<BitCastOp>(Loc,
                                       Function,
                                       C.getPointerType(CallType));
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
          Argument = emitImplicitBitcast(Loc,
                                         Argument,
                                         C.getModelPointerType(T));
          Argument = Builder.create<IndirectionOp>(Loc, T, Argument);
        } else {
          Argument = emitImplicitBitcast(Loc, Argument, T);
        }

        Arguments.push_back(Argument);
      }

      revng_log(ExpressionLog, "CallOp");
      mlir::Value Result = Builder.create<CallOp>(Loc,
                                                  CallType.getReturnType(),
                                                  Function,
                                                  Arguments);

      if (Layout.returnMethod() == abi::FunctionType::ReturnMethod::Scalar) {
        Result = emitImplicitBitcast(SurroundingLocation,
                                     Result,
                                     C.importLLVMType(I->getType()));
      }

      rc_return Result;
    }

    if (auto I = llvm::dyn_cast<llvm::SelectInst>(V)) {
      revng_log(ExpressionLog, "llvm::SelectInst");
      LoggerIndent Indent(ExpressionLog);

      mlir::Location Loc = C.getLocation(I);

      revng_log(ExpressionLog, "Condition subexpression:");
      mlir::Value Condition = (LoggerIndent(ExpressionLog),
                               rc_recur emitExpression(I->getCondition(), Loc));

      revng_log(ExpressionLog, "True subexpression:");
      mlir::Value True = (LoggerIndent(ExpressionLog),
                          rc_recur emitExpression(I->getTrueValue(), Loc));

      revng_log(ExpressionLog, "False subexpression:");
      mlir::Value False = (LoggerIndent(ExpressionLog),
                           rc_recur emitExpression(I->getFalseValue(), Loc));

      mlir::Type ResultType = removeConst(True.getType());
      if (ResultType != removeConst(False.getType())) {
        ResultType = C.importLLVMType(I->getType());
        True = emitImplicitBitcast(Loc, True, ResultType);
        False = emitImplicitBitcast(Loc, False, ResultType);
      }

      rc_return Builder.create<TernaryOp>(Loc,
                                          ResultType,
                                          Condition,
                                          True,
                                          False);
    }

    if (auto *GEP = llvm::dyn_cast<llvm::GetElementPtrInst>(V)) {
      const llvm::Value *GEPPointerOp = GEP->getPointerOperand();
      mlir::Location Loc = C.getLocation(GEP);

      if (GEP->getNumIndices() == 1
          and GEP->getSourceElementType()->isIntegerTy(8)) {

        auto UnsignedCharType = C.getIntegerType(1, IntegerKind::Unsigned);
        auto PointerType = C.getPointerType(UnsignedCharType);

        llvm::Value *GEPIndex = GEP->idx_begin()->get();
        mlir::Value Index = rc_recur emitExpression(GEPIndex, Loc);

        auto IndexType = cast<clift::ValueType>(Index.getType());
        auto IntptrType = cast<clift::ValueType>(C.getIntptrType());
        auto Cmp = IntptrType.getObjectSize() <=> IndexType.getObjectSize();
        if (Cmp < 0)
          Index = emitCast<TruncateOp>(Loc, Index, C.getIntptrType());
        else if (Cmp > 0)
          Index = emitCast<ExtendOp>(Loc, Index, C.getIntptrType());

        mlir::Value
          Pointer = emitCast<BitCastOp>(Loc,
                                        rc_recur emitExpression(GEPPointerOp,
                                                                Loc),
                                        PointerType);
        rc_return emitExpr<PtrAddOp>(Loc, PointerType, Pointer, Index);
      }
    }

    if (auto I = llvm::dyn_cast<llvm::FreezeInst>(V))
      rc_return emitExpression(I->getOperand(0), C.getLocation(I));

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

  //===--------------------- LLVM control flow import ---------------------===//

  template<typename OpT, typename... ArgsT>
  OpT emitLocalDeclaration(mlir::Location Loc, ArgsT &&...Args) {
    mlir::OpBuilder::InsertionGuard Guard(Builder);
    Builder.setInsertionPointToEnd(&LocalDeclarationBlock);
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
      auto Label = emitLocalDeclaration<MakeLabelOp>(C.getLocation(BB));
      Label.setHandle(pipeline::locationString(revng::ranks::GotoLabel,
                                               ModelFunction.Entry(),
                                               NextGotoLabelIndex++));
      Iterator->second.Label = Label;
    }

    if (not Iterator->second.HasAssignLabel
        and Iterator->second.InsertPoint.isSet()) {
      mlir::OpBuilder::InsertionGuard Guard(Builder);
      restoreInsertionPointAfter(Builder, Iterator->second.InsertPoint);
      emitAssignLabel(Iterator->second.Label, C.getLocation(BB));
      Iterator->second.HasAssignLabel = true;
    }

    Builder.create<GotoOp>(Loc, Iterator->second.Label);
  }

  bool isCallToSPTAR(const llvm::CallInst *Call) {
    if (not Call->hasMetadata(PrototypeMDName))
      return false;

    const auto *ModelCallType = getCallSitePrototype(C.Model, Call);
    auto Layout = abi::FunctionType::Layout::make(*ModelCallType);
    namespace ReturnMethod = abi::FunctionType::ReturnMethod;

    return Layout.hasSPTAR();
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
        emitAssignLabel(Iterator->second.Label, C.getLocation(BB));
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
      if (auto *A = llvm::dyn_cast<llvm::AllocaInst>(&I)) {
        const auto *NumElements = A->getArraySize();

        // Non-constant alloca is not supported:
        revng_assert(not NumElements
                     or llvm::isa<llvm::ConstantInt>(NumElements));

        std::optional<std::string> Handle;

        mlir::Type Type;
        if (hasStackFrameMetadata(A)) {
          auto StackType = ModelFunction.stackFrameType();
          revng_assert(StackType);
          Type = C.importType<clift::StructType>(*StackType);
          Handle = pipeline::locationString(revng::ranks::StackFrameVariable,
                                            ModelFunction.Entry());
        } else {
          revng_assert(*A->getAllocationSizeInBits(*C.DataLayout) % 8 == 0);
          llvm::Type *Allocated = A->getAllocatedType();
          revng_assert(Allocated->isSized());

          if (not Allocated->isArrayTy() and not A->isArrayAllocation()) {
            Type = C.importLLVMType(Allocated);

          } else {
            uint64_t FieldBitSize = C.DataLayout->getTypeSizeInBits(Allocated);
            revng_assert(FieldBitSize % 8 == 0);
            uint64_t ElementSizeInBytes = FieldBitSize / 8;
            const auto
              *NumElements = cast<llvm::ConstantInt>(A->getArraySize());
            uint64_t NBytes = NumElements->getZExtValue() * ElementSizeInBytes;
            auto *ByteType = llvm::IntegerType::get(A->getContext(), 8);
            auto *ByteArrayType = llvm::ArrayType::get(ByteType, NBytes);
            Type = C.importLLVMType(ByteArrayType);
          }
        }

        mlir::Location Loc = C.getLocation(A);
        auto Op = emitLocalDeclaration<LocalVariableOp>(Loc, Type);

        if (not Handle) {
          // For any local variables without a more specific handle (e.g. stack
          // frame variable), a generic handle with increasing indices is used.
          Handle = pipeline::locationString(revng::ranks::LocalVariable,
                                            ModelFunction.Entry(),
                                            NextLocalVariableIndex++);
        }

        Op.setHandle(*Handle);

        auto [Iterator, Inserted] = AllocaMapping.try_emplace(A, Op);
        revng_assert(Inserted);

        // Create a clift.require operation to indicate the position dominating
        // all uses of this local variable. This prevents the local variable
        // from being later moved past this point during scope tightening.
        Builder.create<RequireOp>(Loc, Op);

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
        if (isCallToSPTAR(Call)) {
          mlir::Location Loc = C.getLocation(Call);

          auto Op = Builder.create<ExpressionStatementOp>(Loc);

          mlir::Value Local;
          emitExpressionTreeImpl(Op.getExpression(), [&]() {
            mlir::Value Initializer = emitExpression(Call, Loc);

            mlir::Type Type = Initializer.getType();
            Local = emitLocalDeclaration<LocalVariableOp>(Loc, Type);

            return Builder.create<AssignOp>(Loc, Type, Local, Initializer);
          });

          auto [_, Inserted] = SPTARCallVariableMapping.try_emplace(Call,
                                                                    Local);
          revng_assert(Inserted);
          continue;
        }
      }

      // Finally, any instruction with no uses represents the root of an
      // expression tree. Any such tree is emitted as an expression statement.
      if (I.use_empty()) {
        mlir::Location Loc = C.getLocation(&I);
        auto Op = Builder.create<ExpressionStatementOp>(Loc);
        emitExpressionTree(Op.getExpression(), &I, Loc);
      }
    }

    // After the main basic block body is processed, the terminal instruction is
    // handled:

    // Scope-graph goto-markers should only be present in basic blocks
    // terminating with an unconditional branch instruction.
    revng_assert(llvm::isa<llvm::BranchInst>(Terminal) or not HasGotoMarker);
    mlir::Location TerminalLoc = C.getLocation(Terminal);

    if (llvm::isa<llvm::UnreachableInst>(Terminal)) {
      // Do nothing.
    } else if (auto *Return = llvm::dyn_cast<llvm::ReturnInst>(Terminal)) {
      auto Op = Builder.create<ReturnOp>(TerminalLoc);

      if (const llvm::Value *Value = Return->getReturnValue()) {
        clift::FunctionType FunctionType = Function.getFunctionType();

        mlir::Type FuncReturnType = FunctionType.getReturnType();
        mlir::Type LLVMReturnType = FuncReturnType;

        // In SPTAR functions, values are returned by address. In this case
        if (FunctionLayout.hasSPTAR())
          LLVMReturnType = C.getPointerType(LLVMReturnType);

        // Emit the expression tree rooted at the return instruction directly
        // into the expression region of the newly created return operation:
        emitExpressionTreeImpl(Op.getResult(), [&]() {
          mlir::Value ReturnValue = emitExpression(Value, TerminalLoc);

          // In Clift, arrays cannot be returned, so in the case that the
          // returned expression has array type (knowing that an array value
          // must be an lvalue), we can take its address, and read it as the
          // appropriate type (type punning, violates strict aliasing).
          if (auto AT = mlir::dyn_cast<ArrayType>(ReturnValue.getType())) {
            ReturnValue = //
              emitCast<DecayOp>(TerminalLoc,
                                ReturnValue,
                                C.getPointerType(AT.getElementType()));

            ReturnValue = //
              emitCast<BitCastOp>(TerminalLoc,
                                  ReturnValue,
                                  C.getPointerType(LLVMReturnType));

            ReturnValue = Builder.create<IndirectionOp>(TerminalLoc,
                                                        LLVMReturnType,
                                                        ReturnValue);
          }

          if (FunctionLayout.hasSPTAR()) {
            // TODO: This may happen when the pointer size of the Model doesn't
            // match the pointer size on LLVM IR, due to mismatching DataLayout.
            // SPTAR functions return model-pointer-sized integers, with the
            // semantic is to actually return the pointee by copy.
            // Given that the return value is model-pointer-sized, it's size
            // doesn't necessarily match the pointer size in LLVM's DataLayout.
            // Until we don't solve the broader issue of LLVM's DataLayout
            // mismatching the pointer size of the input binary and the Model,
            // we'll have to deal with this corner case.
            // Another option would be to change SPTAR function to return
            // llvm-pointer-sized integers or even LLVM's pointers, instead of
            // model-pointer-sized integers.
            uint64_t
              LLVMPointerSize = unwrapped_cast<PointerType>(LLVMReturnType)
                                  .getObjectSize();
            uint64_t ModelPointerSize = C.getModelPointerSize();
            uint64_t OperandSize = unwrapped_cast<IntegerType>(ReturnValue
                                                                 .getType())
                                     .getObjectSize();
            revng_assert(ModelPointerSize == OperandSize);
            if (ModelPointerSize != LLVMPointerSize)
              ReturnValue = emitIntegerCast(TerminalLoc,
                                            ReturnValue,
                                            LLVMPointerSize);
          }

          // Emit an implicit cast to the required return type if necessary:
          ReturnValue = emitImplicitBitcast(TerminalLoc,
                                            ReturnValue,
                                            LLVMReturnType);

          // In the case of SPTAR, because in the LLVM IR the return is by
          // address, but in Clift the return is by value as usual, a final
          // indirection is needed to convert the LLVM IR pointer to a value:
          if (FunctionLayout.hasSPTAR()) {
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
          emitGoto(C.getLocation(Terminal), Succ);
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

  //===----------------------- LLVM function import -----------------------===//

  static llvm::ArrayRef<abi::FunctionType::Layout::Argument>
  getLayoutArguments(const abi::FunctionType::Layout &Layout) {
    llvm::ArrayRef LayoutArguments = Layout.Arguments;
    if (Layout.hasSPTAR())
      LayoutArguments = LayoutArguments.drop_front();
    return LayoutArguments;
  }

  void initializeArgumentMapping(const llvm::Function *F) {
    llvm::ArrayRef LayoutArguments = getLayoutArguments(FunctionLayout);
    revng_assert(F->arg_size() == Function.getArgumentTypes().size());
    revng_assert(F->arg_size() == LayoutArguments.size());

    // Initialize the argument mapping for the function arguments. By-address
    // parameters are detected using the function layout and require indirection
    // before use. In Clift each argument has the type specified in the model.
    // To make the uses of it as imported from the LLVM IR valid, on each use it
    // must be converted to the type imported directly from the LLVM IR argument
    // type. That type is imported here and stored in the argument description.
    for (const auto [A, T, L] :
         llvm::zip(F->args(), Function.getArgumentTypes(), LayoutArguments)) {
      ArgumentMapping
        .try_emplace(&A,
                     getBodyBlock().addArgument(T, Function->getLoc()),
                     /*TakeAddressOnUse=*/isByAddressParameter(L),
                     /*CastType=*/C.importLLVMType(A.getType()));
    }
  }

  mlir::Block &getBodyBlock() {
    revng_assert(Function);
    revng_assert(Function.getBody().hasOneBlock());
    return Function.getBody().getBlocks().front();
  }

  clift::FunctionOp import(const llvm::Function *F) {
    revng_assert(Function.getBody().empty());
    mlir::Block &BodyBlock = Function.getBody().emplaceBlock();
    Builder.setInsertionPointToEnd(&BodyBlock);

    initializeArgumentMapping(F);

    // Compute the scope graph. While the computation should not modify the
    // function IR, it is not const-correct and so a const_cast must be used.
    PostDomTree.recalculate(const_cast<llvm::Function &>(*F));

    // Finally emit the function body starting at the entry block.
    emitScope(&F->getEntryBlock(), PostDomTree.getRootNode()->getBlock());

    BodyBlock.getOperations().splice(BodyBlock.getOperations().begin(),
                                     LocalDeclarationBlock.getOperations());

    return Function;
  }
};

clift::FunctionOp ClifterImpl::import(const llvm::Function *F) {
  return FunctionClifter::import(*this, F);
}

} // namespace

std::unique_ptr<Clifter> Clifter::make(mlir::ModuleOp Module,
                                       const model::Binary &Model) {
  return std::make_unique<ClifterImpl>(Module, Model);
}
