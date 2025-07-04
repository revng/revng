//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <set>

#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/TypeSwitch.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"

#include "revng/mlir/Dialect/Clift/IR/Clift.h"
#include "revng/mlir/Dialect/Clift/IR/CliftAttributes.h"
#include "revng/mlir/Dialect/Clift/IR/CliftInterfaces.h"
#include "revng/mlir/Dialect/Clift/IR/CliftTypes.h"

#include "CliftBytecode.h"

// This include should stay here for correct build procedure
//
#define GET_ATTRDEF_CLASSES
#include "revng/mlir/Dialect/Clift/IR/CliftAttributes.cpp.inc"

using namespace mlir::clift;
namespace clift = mlir::clift;

using EmitErrorType = llvm::function_ref<mlir::InFlightDiagnostic()>;

//===-------------------------- Class attributes --------------------------===//

using WalkAttrT = llvm::function_ref<void(mlir::Attribute)>;
using WalkTypeT = llvm::function_ref<void(mlir::Type)>;

using ReplaceAttrT = llvm::ArrayRef<mlir::Attribute>;
using ReplaceTypeT = llvm::ArrayRef<mlir::Type>;

namespace mlir::clift {

class ClassAttrStorage : public mlir::AttributeStorage {
  struct Key {
    llvm::StringRef Handle;
    std::optional<ClassDefinition> Definition;

    explicit Key(llvm::StringRef Handle) : Handle(Handle) {}

    friend bool operator==(const Key &LHS, const Key &RHS) {
      return LHS.Handle == RHS.Handle;
    }

    [[nodiscard]] llvm::hash_code hashValue() const {
      return llvm::hash_value(Handle);
    }
  };

  Key TheKey;

public:
  using KeyTy = Key;

  const Key &getAsKey() const { return TheKey; }

  static llvm::hash_code hashKey(const Key &Key) { return Key.hashValue(); }

  friend bool operator==(const ClassAttrStorage &LHS, const Key &RHS) {
    return LHS.TheKey == RHS;
  }

  explicit ClassAttrStorage(llvm::StringRef Handle) : TheKey(Handle) {}

  static ClassAttrStorage *
  construct(mlir::StorageUniquer::StorageAllocator &Allocator, const Key &Key) {
    void *Storage = Allocator.allocate<ClassAttrStorage>();
    llvm::StringRef Handle = Allocator.copyInto(Key.Handle);
    auto *S = new (Storage) ClassAttrStorage(Handle);
    S->TheKey.Definition = Key.Definition;
    return S;
  }

  mlir::LogicalResult mutate(mlir::StorageUniquer::StorageAllocator &Allocator,
                             const ClassDefinition &Definition) {
    if (TheKey.Definition)
      return mlir::success(Definition == TheKey.Definition);

    TheKey.Definition.emplace(Allocator.copyInto(Definition.Name),
                              Definition.Size,
                              Allocator.copyInto(Definition.Fields));

    return mlir::success();
  }

  llvm::StringRef getHandle() const { return TheKey.Handle; }

  const ClassDefinition *getDefinitionOrNull() const {
    return TheKey.Definition ? &*TheKey.Definition : nullptr;
  }

  const ClassDefinition &getDefinition() const {
    revng_check(TheKey.Definition);
    return *TheKey.Definition;
  }

  ClassDefinition &getMutableDefinition() {
    revng_check(TheKey.Definition);
    return *TheKey.Definition;
  }
};

template<typename AttrT>
llvm::StringRef ClassAttrImpl<AttrT>::getHandle() const {
  return Base::getImpl()->getHandle();
}

template<typename AttrT>
bool ClassAttrImpl<AttrT>::hasDefinition() const {
  return Base::getImpl()->getDefinitionOrNull() != nullptr;
}

template<typename AttrT>
const ClassDefinition *ClassAttrImpl<AttrT>::getDefinitionOrNull() const {
  return Base::getImpl()->getDefinitionOrNull();
}

template<typename AttrT>
const ClassDefinition &ClassAttrImpl<AttrT>::getDefinition() const {
  return Base::getImpl()->getDefinition();
}

template<typename AttrT>
void ClassAttrImpl<AttrT>::walkImmediateSubElements(WalkAttrT WalkAttr,
                                                    WalkTypeT WalkType) const {
  for (auto Field : getDefinition().getFields())
    WalkAttr(Field);
}

template<typename AttrT>
mlir::Attribute
ClassAttrImpl<AttrT>::replaceImmediateSubElements(ReplaceAttrT NewAttrs,
                                                  ReplaceTypeT NewTypes) const {
  revng_abort("Cannot replace sub-elements of a class attribute.");
}

template class ClassAttrImpl<StructAttr>;
template class ClassAttrImpl<UnionAttr>;

} // namespace mlir::clift

//===------------------------------ FieldAttr -----------------------------===//

mlir::LogicalResult FieldAttr::verify(EmitErrorType EmitError,
                                      uint64_t Offset,
                                      clift::ValueType ElementType,
                                      llvm::StringRef Name) {
  if (not isObjectType(ElementType)) {
    return EmitError() << "Struct and union field types must be object types. "
                       << "Field at offset " << Offset << " is not.";
  }

  return mlir::success();
}

//===---------------------------- EnumFieldAttr ---------------------------===//

mlir::LogicalResult EnumFieldAttr::verify(EmitErrorType EmitError,
                                          uint64_t RawValue,
                                          llvm::StringRef Name) {
  return mlir::success();
}

template<std::same_as<EnumFieldAttr>>
static EnumFieldAttr readAttr(mlir::DialectBytecodeReader &Reader) {
  uint64_t RawValue;
  if (Reader.readVarInt(RawValue).failed())
    return {};

  llvm::StringRef Name;
  if (Reader.readString(Name).failed())
    return {};

  return EnumFieldAttr::get(Reader.getContext(), RawValue, Name);
}

static void writeAttr(EnumFieldAttr Attr, mlir::DialectBytecodeWriter &Writer) {
  Writer.writeVarInt(Attr.getRawValue());
  Writer.writeOwnedString(Attr.getName());
}

//===------------------------------ EnumAttr ------------------------------===//

mlir::LogicalResult EnumAttr::verify(EmitErrorType EmitError,
                                     llvm::StringRef Handle,
                                     llvm::StringRef Name,
                                     clift::ValueType UnderlyingType,
                                     llvm::ArrayRef<EnumFieldAttr> Fields) {
  auto [DealiasedType, HasConst] = decomposeTypedef(UnderlyingType);

  auto PrimitiveType = mlir::dyn_cast<clift::PrimitiveType>(DealiasedType);
  if (not PrimitiveType or HasConst or PrimitiveType.isConst())
    return EmitError() << "Underlying type of enum must be a non-const "
                          "primitive type";

  const uint64_t BitWidth = PrimitiveType.getSize() * 8;

  if (Fields.empty())
    return EmitError() << "enum requires at least one field";

  uint64_t MinValue = 0;
  uint64_t MaxValue = 0;
  bool IsSigned = false;

  switch (PrimitiveType.getKind()) {
  case PrimitiveKind::UnsignedKind:
    MaxValue = llvm::APInt::getMaxValue(BitWidth).getZExtValue();
    break;
  case PrimitiveKind::SignedKind:
    MinValue = llvm::APInt::getSignedMinValue(BitWidth).getSExtValue();
    MaxValue = llvm::APInt::getSignedMaxValue(BitWidth).getSExtValue();
    IsSigned = true;
    break;
  default:
    return EmitError() << "enum underlying type must be an integral type";
  }

  uint64_t LastValue = 0;
  bool CheckEqual = false;

  for (const auto &Field : Fields) {
    const uint64_t Value = Field.getRawValue();

    const auto UsingSigned = [&](auto Callable, const auto... V) {
      return IsSigned ? Callable(static_cast<int64_t>(V)...) : Callable(V...);
    };

    const auto CheckSigned =
      [EmitError](const auto Value,
                  const auto MinValue,
                  const auto MaxValue) -> mlir::LogicalResult {
      if (Value < MinValue)
        return EmitError() << "enum field " << Value
                           << " is less than the min value of the "
                              "underlying type "
                           << MinValue;

      if (Value > MaxValue)
        return EmitError() << "enum field " << Value
                           << " is greater than the max value of the "
                              "underlying type "
                           << MaxValue;

      return mlir::success();
    };

    const mlir::LogicalResult R = UsingSigned(CheckSigned,
                                              Value,
                                              MinValue,
                                              MaxValue);

    if (failed(R))
      return R;

    if (Value < LastValue || (CheckEqual && Value == LastValue))
      return EmitError() << "enum fields must be strictly ordered by their "
                            "unsigned values";

    LastValue = Value;
    CheckEqual = true;
  }

  return mlir::success();
}

template<std::same_as<EnumAttr>>
static EnumAttr readAttr(mlir::DialectBytecodeReader &Reader) {
  llvm::StringRef Handle;
  if (Reader.readString(Handle).failed())
    return {};

  llvm::StringRef Name;
  if (Reader.readString(Name).failed())
    return {};

  clift::ValueType UnderlyingType;
  if (Reader.readType(UnderlyingType).failed())
    return {};

  auto ReadField = [&](EnumFieldAttr &Field) {
    return Reader.readAttribute(Field);
  };

  llvm::SmallVector<EnumFieldAttr> Fields;
  if (Reader.readList(Fields, ReadField).failed())
    return {};

  return EnumAttr::get(Reader.getContext(),
                       Handle,
                       Name,
                       UnderlyingType,
                       std::move(Fields));
}

static void writeAttr(EnumAttr Attr, mlir::DialectBytecodeWriter &Writer) {
  Writer.writeOwnedString(Attr.getHandle());
  Writer.writeOwnedString(Attr.getName());
  Writer.writeType(Attr.getUnderlyingType());
  Writer.writeList(Attr.getFields(), [&](EnumFieldAttr Field) {
    return Writer.writeAttribute(Field);
  });
}

//===----------------------------- TypedefAttr ----------------------------===//

mlir::LogicalResult TypedefAttr::verify(EmitErrorType EmitError,
                                        llvm::StringRef Handle,
                                        llvm::StringRef Name,
                                        clift::ValueType UnderlyingType) {
  return mlir::success();
}

template<std::same_as<TypedefAttr>>
static TypedefAttr readAttr(mlir::DialectBytecodeReader &Reader) {
  llvm::StringRef Handle;
  if (Reader.readString(Handle).failed())
    return {};

  llvm::StringRef Name;
  if (Reader.readString(Name).failed())
    return {};

  clift::ValueType UnderlyingType;
  if (Reader.readType(UnderlyingType).failed())
    return {};

  return TypedefAttr::get(Reader.getContext(), Handle, Name, UnderlyingType);
}

static void writeAttr(TypedefAttr Attr, mlir::DialectBytecodeWriter &Writer) {
  Writer.writeOwnedString(Attr.getHandle());
  Writer.writeOwnedString(Attr.getName());
  Writer.writeType(Attr.getUnderlyingType());
}

//===----------------------------- StructAttr -----------------------------===//

mlir::LogicalResult StructAttr::verify(EmitErrorType EmitError,
                                       llvm::StringRef Handle) {
  return mlir::success();
}

mlir::LogicalResult StructAttr::verify(EmitErrorType EmitError,
                                       llvm::StringRef Handle,
                                       const ClassDefinition &Definition) {
  return mlir::success();
}

mlir::LogicalResult StructAttr::verify(EmitErrorType EmitError,
                                       llvm::StringRef Handle,
                                       llvm::StringRef Name,
                                       uint64_t Size,
                                       llvm::ArrayRef<FieldAttr> Fields) {
  return mlir::success();
}

mlir::LogicalResult
StructAttr::verifyDefinition(EmitErrorType EmitError) const {
  const ClassDefinition &Definition = getDefinition();

  if (Definition.getSize() == 0)
    return EmitError() << "struct type cannot have a size of zero";

  if (not Definition.getFields().empty()) {
    uint64_t LastEndOffset = 0;

    llvm::SmallSet<llvm::StringRef, 16> NameSet;
    for (const auto &Field : Definition.getFields()) {
      if (Field.getOffset() < LastEndOffset)
        return EmitError() << "Fields of structs must be ordered by offset, "
                              "and "
                              "they cannot overlap";

      LastEndOffset = Field.getOffset() + Field.getType().getByteSize();

      if (not Field.getName().empty()) {
        if (not NameSet.insert(Field.getName()).second)
          return EmitError() << "struct field names must be empty or unique";
      }
    }

    if (LastEndOffset > Definition.getSize())
      return EmitError() << "offset + size of field of struct type is greater "
                            "than the struct type size.";
  }

  return mlir::success();
}

StructAttr StructAttr::get(MLIRContext *Context, llvm::StringRef Handle) {
  return Base::get(Context, Handle);
}

StructAttr StructAttr::getChecked(EmitErrorType EmitError,
                                  MLIRContext *Context,
                                  llvm::StringRef Handle) {
  return Base::get(Context, Handle);
}

StructAttr StructAttr::get(MLIRContext *Context,
                           llvm::StringRef Handle,
                           const ClassDefinition &Definition) {
  auto Attr = Base::get(Context, Handle);
  auto R = Attr.Base::mutate(Definition);
  revng_assert(R.succeeded()
               and "Attempted to mutate the definition of an already defined "
                   "struct attribute.");
  return Attr;
}

StructAttr StructAttr::getChecked(EmitErrorType EmitError,
                                  MLIRContext *Context,
                                  llvm::StringRef Handle,
                                  const ClassDefinition &Definition) {
  return get(Context, Handle, Definition);
}

StructAttr StructAttr::get(MLIRContext *Context,
                           llvm::StringRef Handle,
                           llvm::StringRef Name,
                           uint64_t Size,
                           llvm::ArrayRef<FieldAttr> Fields) {
  return get(Context, Handle, ClassDefinition{ Name, Size, Fields });
}

StructAttr StructAttr::getChecked(EmitErrorType EmitError,
                                  MLIRContext *Context,
                                  llvm::StringRef Handle,
                                  llvm::StringRef Name,
                                  uint64_t Size,
                                  llvm::ArrayRef<FieldAttr> Fields) {
  return getChecked(EmitError,
                    Context,
                    Handle,
                    ClassDefinition{ Name, Size, Fields });
}

//===------------------------------ UnionAttr -----------------------------===//

static uint64_t getUnionSize(llvm::ArrayRef<FieldAttr> Fields) {
  uint64_t Max = 0;
  for (auto const &Field : Fields)
    Max = std::max(Max, Field.getType().getByteSize());
  return Max;
}

mlir::LogicalResult UnionAttr::verify(EmitErrorType EmitError,
                                      llvm::StringRef Handle) {
  return mlir::success();
}

mlir::LogicalResult UnionAttr::verify(EmitErrorType EmitError,
                                      llvm::StringRef Handle,
                                      const ClassDefinition &Definition) {
  return mlir::success();
}

mlir::LogicalResult UnionAttr::verify(EmitErrorType EmitError,
                                      llvm::StringRef Handle,
                                      llvm::StringRef Name,
                                      llvm::ArrayRef<FieldAttr> Fields) {
  return mlir::success();
}

mlir::LogicalResult UnionAttr::verifyDefinition(EmitErrorType EmitError) const {
  const ClassDefinition &Definition = getDefinition();

  if (Definition.getFields().empty())
    return EmitError() << "union types must have at least one field";

  llvm::SmallSet<llvm::StringRef, 16> NameSet;
  for (const auto &Field : Definition.getFields()) {
    if (Field.getOffset() != 0)
      return EmitError() << "union field offsets must be zero";

    if (not Field.getName().empty()) {
      if (not NameSet.insert(Field.getName()).second)
        return EmitError() << "union field names must be empty or unique";
    }
  }

  return mlir::success();
}

UnionAttr UnionAttr::get(MLIRContext *Context, llvm::StringRef Handle) {
  return Base::get(Context, Handle);
}

UnionAttr UnionAttr::getChecked(EmitErrorType EmitError,
                                MLIRContext *Context,
                                llvm::StringRef Handle) {
  return Base::get(Context, Handle);
}

UnionAttr UnionAttr::get(MLIRContext *Context,
                         llvm::StringRef Handle,
                         const ClassDefinition &Definition) {
  auto Attr = Base::get(Context, Handle);
  auto R = Attr.Base::mutate(Definition);
  revng_assert(R.succeeded()
               and "Attempted to mutate the definition of an already defined "
                   "union attribute.");
  return Attr;
}

UnionAttr UnionAttr::getChecked(EmitErrorType EmitError,
                                MLIRContext *Context,
                                llvm::StringRef Handle,
                                const ClassDefinition &Definition) {
  return get(Context, Handle, Definition);
}

UnionAttr UnionAttr::get(MLIRContext *Context,
                         llvm::StringRef Handle,
                         llvm::StringRef Name,
                         llvm::ArrayRef<FieldAttr> Fields) {
  return get(Context, Handle, ClassDefinition{ Name, 0, Fields });
}

UnionAttr UnionAttr::getChecked(EmitErrorType EmitError,
                                MLIRContext *Context,
                                llvm::StringRef Handle,
                                llvm::StringRef Name,
                                llvm::ArrayRef<FieldAttr> Fields) {
  return getChecked(EmitError,
                    Context,
                    Handle,
                    ClassDefinition{ Name, 0, Fields });
}

uint64_t UnionAttr::getSize() const {
  ClassDefinition &Definition = Base::getImpl()->getMutableDefinition();

  uint64_t Size = Definition.Size;
  if (Size == 0) {
    // Technically since this is a const member function, another thread could
    // be concurrently observing the zero size and mutating the same object.
    // While this is technically UB (std::atomic_ref should be used instead but
    // is not yet available), it should not be a problem because both threads
    // are expected to compute the same value, and the shared object is only
    // used for caching and not synchronisation.
    Definition.Size = Size = getUnionSize(Definition.Fields);
  }
  return Size;
}

//===---------------------------- CliftDialect ----------------------------===//

void CliftDialect::registerAttributes() {
  addAttributes<StructAttr, UnionAttr,
  // Include the list of auto-generated attributes
#define GET_ATTRDEF_LIST
#include "revng/mlir/Dialect/Clift/IR/CliftAttributes.cpp.inc"
                /* End of auto-generated list */>();
}

/// Parse an attribute registered to this dialect
mlir::Attribute CliftDialect::parseAttribute(mlir::DialectAsmParser &Parser,
                                             mlir::Type Type) const {
  return {};
}

/// Print an attribute registered to this dialect
void CliftDialect::printAttribute(mlir::Attribute Attr,
                                  mlir::DialectAsmPrinter &Printer) const {
  revng_abort("cannot print attribute");
}

namespace {

enum class CliftAttrKind : uint8_t {
  Typedef,
  EnumField,
  Enum,
  Struct,
  Union,

  N
};

} // namespace

static mlir::LogicalResult readAttrKind(CliftAttrKind &TypeKind,
                                        mlir::DialectBytecodeReader &Reader) {
  uint64_t Value;
  if (Reader.readVarInt(Value).failed())
    return mlir::failure();

  if (Value >= static_cast<uint64_t>(CliftAttrKind::N))
    return mlir::failure();

  TypeKind = static_cast<CliftAttrKind>(Value);
  return mlir::success();
}

mlir::Attribute clift::readAttr(mlir::DialectBytecodeReader &Reader) {
  CliftAttrKind TypeKind;
  if (readAttrKind(TypeKind, Reader).failed())
    return {};

  switch (TypeKind) {
  case CliftAttrKind::Typedef:
    return ::readAttr<clift::TypedefAttr>(Reader);
  case CliftAttrKind::EnumField:
    return ::readAttr<clift::EnumFieldAttr>(Reader);
  case CliftAttrKind::Enum:
    return ::readAttr<clift::EnumAttr>(Reader);
  case CliftAttrKind::Struct:
    return BytecodeClassAttr::get(Reader.getContext(),
                                  clift::readStructDefinition(Reader));
  case CliftAttrKind::Union:
    return BytecodeClassAttr::get(Reader.getContext(),
                                  clift::readUnionDefinition(Reader));
  case CliftAttrKind::N:
    break;
  }
  revng_abort();
}

mlir::LogicalResult clift::writeAttr(mlir::Attribute Attr,
                                     mlir::DialectBytecodeWriter &Writer) {
  auto WriteKind = [&](CliftAttrKind TypeKind) {
    Writer.writeVarInt(static_cast<uint64_t>(TypeKind));
  };

  auto Write = [&](auto T, CliftAttrKind TypeKind) {
    WriteKind(TypeKind);
    ::writeAttr(T, Writer);
    return mlir::success();
  };

  if (auto A = mlir::dyn_cast<clift::TypedefAttr>(Attr))
    return Write(A, CliftAttrKind::Typedef);
  if (auto A = mlir::dyn_cast<clift::EnumFieldAttr>(Attr))
    return Write(A, CliftAttrKind::EnumField);
  if (auto A = mlir::dyn_cast<clift::EnumAttr>(Attr))
    return Write(A, CliftAttrKind::Enum);

  if (auto A = mlir::dyn_cast<clift::BytecodeClassAttr>(Attr)) {
    if (auto T = mlir::dyn_cast<clift::StructType>(A.getType())) {
      WriteKind(CliftAttrKind::Struct);
      writeStructDefinition(T, Writer);
      return mlir::success();
    }
    if (auto T = mlir::dyn_cast<clift::UnionType>(A.getType())) {
      WriteKind(CliftAttrKind::Union);
      writeUnionDefinition(T, Writer);
      return mlir::success();
    }
  }

  return mlir::failure();
}
