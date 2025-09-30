//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <set>

#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/StringExtras.h"
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

//===----------------------- Implementation helpers -----------------------===//

static auto getEmitError(mlir::AsmParser &Parser, const mlir::SMLoc &Location) {
  return [&Parser, Location]() { return Parser.emitError(Location); };
}

static void printString(mlir::AsmPrinter &Printer, llvm::StringRef String) {
  Printer << '\"';
  llvm::printEscapedString(String, Printer.getStream());
  Printer << '\"';
}

//===-------------------------- Class attributes --------------------------===//

using WalkAttrT = llvm::function_ref<void(mlir::Attribute)>;
using WalkTypeT = llvm::function_ref<void(mlir::Type)>;

using ReplaceAttrT = llvm::ArrayRef<mlir::Attribute>;
using ReplaceTypeT = llvm::ArrayRef<mlir::Type>;

namespace mlir::clift {

class MutableStringAttrStorage : public mlir::AttributeStorage {
  struct Pair {
    mlir::Attribute Key;
    llvm::StringRef Value;

    explicit Pair(mlir::Attribute Key) : Key(Key) {}

    friend bool operator==(const Pair &LHS, const Pair &RHS) {
      return LHS.Key.getAsOpaquePointer() == RHS.Key.getAsOpaquePointer();
    }

    [[nodiscard]] llvm::hash_code hashValue() const {
      return llvm::hash_value(Key.getAsOpaquePointer());
    }
  };

  Pair TheKey;

public:
  using KeyTy = Pair;

  const Pair &getAsKey() const { return TheKey; }

  static llvm::hash_code hashKey(const Pair &Key) { return Key.hashValue(); }

  friend bool operator==(const MutableStringAttrStorage &LHS, const Pair &RHS) {
    return LHS.TheKey == RHS;
  }

  explicit MutableStringAttrStorage(mlir::Attribute Key) : TheKey(Key) {}

  static MutableStringAttrStorage *
  construct(mlir::StorageUniquer::StorageAllocator &Allocator,
            const Pair &Key) {
    void *Storage = Allocator.allocate<MutableStringAttrStorage>();
    auto *S = new (Storage) MutableStringAttrStorage(Key.Key);
    return S;
  }

  mlir::LogicalResult mutate(mlir::StorageUniquer::StorageAllocator &Allocator,
                             llvm::StringRef Value) {
    TheKey.Value = Allocator.copyInto(Value);
    return mlir::success();
  }

  mlir::Attribute getKey() const { return TheKey.Key; }
  llvm::StringRef getValue() const { return TheKey.Value; }
};

MutableStringAttr MutableStringAttr::get(mlir::MLIRContext *Context,
                                         mlir::Attribute Key) {
  return Base::get(Context, Key);
}

MutableStringAttr MutableStringAttr::get(mlir::MLIRContext *Context,
                                         mlir::Attribute Key,
                                         llvm::StringRef Value) {
  auto Attr = get(Context, Key);
  Attr.setValue(Value);
  return Attr;
}

mlir::Attribute MutableStringAttr::getKey() const {
  return getImpl()->getKey();
}

llvm::StringRef MutableStringAttr::getValue() const {
  return getImpl()->getValue();
}

void MutableStringAttr::setValue(llvm::StringRef Value) {
  (void) Base::mutate(Value);
}

void MutableStringAttr::walkImmediateSubElements(WalkAttrT WalkAttrs,
                                                 WalkTypeT WalkTypes) const {
  WalkAttrs(getImpl()->getKey());
}

Attribute
MutableStringAttr::replaceImmediateSubElements(ReplaceAttrT NewAttrs,
                                               ReplaceTypeT NewTypes) const {
  revng_assert(NewAttrs.size() == 1);
  revng_assert(NewTypes.size() == 0);
  return MutableStringAttr::get(getContext(), NewAttrs.front());
}

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

    TheKey.Definition.emplace(Definition.Name,
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

//===---------------------------- AttributeAttr ---------------------------===//

static mlir::LogicalResult parseAttributeComponent(mlir::AsmParser &Parser,
                                                   AttributeComponentAttr &C) {
  mlir::SMLoc Loc = Parser.getCurrentLocation();

  std::string String;
  if (Parser.parseString(&String).failed())
    return mlir::failure();

  std::string Handle;
  if (Parser.parseOptionalColon().succeeded()) {
    if (Parser.parseString(&Handle).failed())
      return mlir::failure();
  }

  C = AttributeComponentAttr::getChecked(getEmitError(Parser, Loc),
                                         Parser.getContext(),
                                         String,
                                         Handle);

  return mlir::success();
}

mlir::Attribute AttributeAttr::parse(mlir::AsmParser &Parser, mlir::Type Type) {
  mlir::SMLoc Loc = Parser.getCurrentLocation();

  if (Parser.parseLess().failed())
    return {};

  AttributeComponentAttr Macro;
  if (parseAttributeComponent(Parser, Macro).failed())
    return {};

  llvm::SmallVector<AttributeComponentAttr> ArgumentsArray;
  std::optional<llvm::ArrayRef<AttributeComponentAttr>> Arguments;

  if (Parser.parseOptionalLParen().succeeded()) {
    if (Parser.parseOptionalRParen().failed()) {
      auto ParseArgument = [&Parser, &ArgumentsArray]() -> mlir::ParseResult {
        AttributeComponentAttr Argument;
        if (parseAttributeComponent(Parser, Argument).failed())
          return mlir::failure();

        ArgumentsArray.push_back(Argument);
        return mlir::success();
      };

      if (Parser.parseCommaSeparatedList(ParseArgument).failed())
        return {};

      if (Parser.parseRParen().failed())
        return {};
    }

    Arguments = ArgumentsArray;
  }

  if (Parser.parseGreater().failed())
    return {};

  return AttributeAttr::getChecked(getEmitError(Parser, Loc),
                                   Parser.getContext(),
                                   Macro,
                                   Arguments);
}

static void printAttributeComponent(mlir::AsmPrinter &Printer,
                                    AttributeComponentAttr C) {
  printString(Printer, C.getString());

  if (not C.getHandle().empty()) {
    Printer << " : ";
    printString(Printer, C.getHandle());
  }
}

void AttributeAttr::print(mlir::AsmPrinter &Printer) const {
  Printer << '<';

  printAttributeComponent(Printer, getMacro());

  if (const auto &Arguments = getArguments()) {
    Printer << '(';
    for (auto [I, A] : llvm::enumerate(*Arguments)) {
      if (I != 0)
        Printer << ", ";

      printAttributeComponent(Printer, A);
    }
    Printer << ')';
  }

  Printer << '>';
}

//===------------------------------ FieldAttr -----------------------------===//

mlir::LogicalResult FieldAttr::verify(EmitErrorType EmitError,
                                      llvm::StringRef Handle,
                                      MutableStringAttr Name,
                                      uint64_t Offset,
                                      clift::ValueType ElementType) {
  if (not isObjectType(ElementType)) {
    return EmitError() << "Struct and union field types must be object types. "
                       << "Field at offset " << Offset << " is not.";
  }

  return mlir::success();
}

//===---------------------------- EnumFieldAttr ---------------------------===//

mlir::LogicalResult EnumFieldAttr::verify(EmitErrorType EmitError,
                                          llvm::StringRef Handle,
                                          MutableStringAttr Name,
                                          uint64_t RawValue) {
  return mlir::success();
}

template<std::same_as<EnumFieldAttr>>
static EnumFieldAttr readAttr(mlir::DialectBytecodeReader &Reader) {
  llvm::StringRef Handle;
  if (Reader.readString(Handle).failed())
    return {};

  llvm::StringRef Name;
  if (Reader.readString(Name).failed())
    return {};

  uint64_t RawValue;
  if (Reader.readVarInt(RawValue).failed())
    return {};

  return EnumFieldAttr::get(Reader.getContext(),
                            Handle,
                            makeNameAttr<EnumFieldAttr>(Reader.getContext(),
                                                        Handle,
                                                        Name),
                            RawValue);
}

static void writeAttr(EnumFieldAttr Attr, mlir::DialectBytecodeWriter &Writer) {
  Writer.writeOwnedString(Attr.getHandle());
  Writer.writeOwnedString(Attr.getName());
  Writer.writeVarInt(Attr.getRawValue());
}

//===------------------------------ EnumAttr ------------------------------===//

mlir::LogicalResult EnumAttr::verify(EmitErrorType EmitError,
                                     llvm::StringRef Handle,
                                     MutableStringAttr Name,
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
                       makeNameAttr<EnumAttr>(Reader.getContext(),
                                              Handle,
                                              Name),
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
                                        MutableStringAttr Name,
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

  return TypedefAttr::get(Reader.getContext(),
                          Handle,
                          makeNameAttr<TypedefAttr>(Reader.getContext(),
                                                    Handle,
                                                    Name),
                          UnderlyingType);
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
                                       MutableStringAttr Name,
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
                           MutableStringAttr Name,
                           uint64_t Size,
                           llvm::ArrayRef<FieldAttr> Fields) {
  return get(Context, Handle, ClassDefinition{ Name, Size, Fields });
}

StructAttr StructAttr::getChecked(EmitErrorType EmitError,
                                  MLIRContext *Context,
                                  llvm::StringRef Handle,
                                  MutableStringAttr Name,
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
                                      MutableStringAttr Name,
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
                         MutableStringAttr Name,
                         llvm::ArrayRef<FieldAttr> Fields) {
  return get(Context, Handle, ClassDefinition{ Name, 0, Fields });
}

UnionAttr UnionAttr::getChecked(EmitErrorType EmitError,
                                MLIRContext *Context,
                                llvm::StringRef Handle,
                                MutableStringAttr Name,
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
  addAttributes<MutableStringAttr, StructAttr, UnionAttr,
  // Include the list of auto-generated attributes
#define GET_ATTRDEF_LIST
#include "revng/mlir/Dialect/Clift/IR/CliftAttributes.cpp.inc"
                /* End of auto-generated list */>();
}

/// Parse an attribute registered to this dialect
mlir::Attribute CliftDialect::parseAttribute(mlir::DialectAsmParser &Parser,
                                             mlir::Type Type) const {
  llvm::StringRef Mnemonic;
  if (mlir::Attribute Attr;
      generatedAttributeParser(Parser, &Mnemonic, Type, Attr).has_value())
    return Attr;

  return {};
}

/// Print an attribute registered to this dialect
void CliftDialect::printAttribute(mlir::Attribute Attr,
                                  mlir::DialectAsmPrinter &Printer) const {
  if (mlir::succeeded(generatedAttributePrinter(Attr, Printer)))
    return;

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
