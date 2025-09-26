//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <optional>
#include <string>

#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/StringExtras.h"

#include "revng/Support/Identifier.h"
#include "revng/mlir/Dialect/Clift/IR/CliftTypes.h"

#include "CliftBytecode.h"

// keep this order
#include "revng/mlir/Dialect/Clift/IR/CliftAttributes.h"

namespace mlir {

static ParseResult parseCliftDebugName(AsmParser &Parser, std::string &Name);
static void printCliftDebugName(AsmPrinter &Printer, llvm::StringRef Name);

} // namespace mlir

#define GET_TYPEDEF_CLASSES
#include "revng/mlir/Dialect/Clift/IR/CliftOpsTypes.cpp.inc"

using namespace mlir::clift;
namespace clift = mlir::clift;

using EmitErrorType = llvm::function_ref<mlir::InFlightDiagnostic()>;

//===----------------------- Implementation helpers -----------------------===//

static auto getEmitError(mlir::AsmParser &Parser, const mlir::SMLoc &Location) {
  return [&Parser, Location]() { return Parser.emitError(Location); };
}

static auto getEmitError(mlir::DialectBytecodeReader &Reader) {
  return [&Reader]() { return Reader.emitError(); };
}

static bool getDefinedTypeAliasImpl(DefinedType Type,
                                    llvm::StringRef Kind,
                                    llvm::raw_ostream &OS) {
  if (not Type.getName().empty()) {
    OS << Type.getName();
  } else if (not Type.getHandle().empty()) {
    OS << sanitizeIdentifier(Type.getHandle());
  } else {
    OS << Kind;
  }

  return true;
}

template<typename DefinedTypeT>
static bool getDefinedTypeAlias(DefinedTypeT Type, llvm::raw_ostream &OS) {
  return getDefinedTypeAliasImpl(Type, DefinedTypeT::getMnemonic(), OS);
}

static void printString(mlir::AsmPrinter &Printer, llvm::StringRef String) {
  Printer << '\"';
  llvm::printEscapedString(String, Printer.getStream());
  Printer << '\"';
}

static mlir::ParseResult mlir::parseCliftDebugName(mlir::AsmParser &Parser,
                                                   std::string &Name) {
  if (Parser.parseOptionalKeyword("as").succeeded()) {
    if (Parser.parseString(&Name).failed())
      return mlir::failure();
  }
  return mlir::success();
}

static void mlir::printCliftDebugName(mlir::AsmPrinter &Printer,
                                      llvm::StringRef Name) {
  if (not Name.empty()) {
    Printer << " as ";
    printString(Printer, Name);
  }
}

static mlir::LogicalResult readBool(bool &Value,
                                    mlir::DialectBytecodeReader &Reader) {
  uint64_t Integer;
  if (Reader.readVarInt(Integer).failed())
    return mlir::failure();
  if (Integer > 1)
    return mlir::failure();
  Value = Integer != 0;
  return mlir::success();
}

static void writeBool(bool Value, mlir::DialectBytecodeWriter &Writer) {
  Writer.writeVarInt(Value);
}

//===------------------------------ LabelType -----------------------------===//

template<std::same_as<clift::LabelType>>
static clift::LabelType readType(mlir::DialectBytecodeReader &Reader) {
  return clift::LabelType::get(Reader.getContext());
}

static void writeType(clift::LabelType Type,
                      mlir::DialectBytecodeWriter &Writer) {
}

//===---------------------------- PrimitiveType ---------------------------===//

mlir::LogicalResult PrimitiveType::verify(EmitErrorType EmitError,
                                          PrimitiveKind Kind,
                                          uint64_t Size,
                                          bool IsConst) {
  if (Kind == PrimitiveKind::VoidKind and Size != 0)
    return EmitError() << "primitive type of void kind must have size of 0.";

  return mlir::success();
}

static llvm::StringRef getPrimitiveKindAlias(PrimitiveKind Kind) {
  switch (Kind) {
  case PrimitiveKind::SignedKind:
    return "int";
  case PrimitiveKind::UnsignedKind:
    return "uint";
  default:
    return stringifyPrimitiveKind(Kind);
  }
}

bool PrimitiveType::getAlias(llvm::raw_ostream &OS) const {
  if (getIsConst())
    return false;

  OS << getPrimitiveKindAlias(getKind());
  if (getKind() != PrimitiveKind::VoidKind)
    OS << getByteSize() * 8 << "_t";

  return true;
}

ValueType PrimitiveType::addConst() const {
  if (isConst())
    return *this;

  return get(getContext(), getKind(), getSize(), /*IsConst=*/true);
}

ValueType PrimitiveType::removeConst() const {
  if (not isConst())
    return *this;

  return get(getContext(), getKind(), getSize(), /*IsConst=*/false);
}

template<std::same_as<clift::PrimitiveType>>
static clift::PrimitiveType readType(mlir::DialectBytecodeReader &Reader) {
  uint64_t KindInteger;
  if (Reader.readVarInt(KindInteger).failed())
    return {};

  auto Kind = symbolizePrimitiveKind(KindInteger);
  if (not Kind)
    return {};

  uint64_t Size;
  if (Reader.readVarInt(Size).failed())
    return {};

  bool Const;
  if (readBool(Const, Reader).failed())
    return {};

  return clift::PrimitiveType::get(Reader.getContext(), *Kind, Size, Const);
}

static void writeType(clift::PrimitiveType Type,
                      mlir::DialectBytecodeWriter &Writer) {
  Writer.writeVarInt(static_cast<uint64_t>(Type.getKind()));
  Writer.writeVarInt(Type.getSize());
  writeBool(Type.getIsConst(), Writer);
}

//===----------------------------- PointerType ----------------------------===//

mlir::LogicalResult PointerType::verify(EmitErrorType EmitError,
                                        ValueType ElementType,
                                        uint64_t PointerSize,
                                        bool IsConst) {
  switch (PointerSize) {
  case 4:
  case 8:
    break;
  default:
    return EmitError() << "invalid pointer size: " << PointerSize;
  }
  return mlir::success();
}

ValueType PointerType::addConst() const {
  if (isConst())
    return *this;

  return get(getContext(),
             getPointeeType(),
             getPointerSize(),
             /*IsConst=*/true);
}

ValueType PointerType::removeConst() const {
  if (not isConst())
    return *this;

  return get(getContext(),
             getPointeeType(),
             getPointerSize(),
             /*IsConst=*/false);
}

template<std::same_as<clift::PointerType>>
static clift::PointerType readType(mlir::DialectBytecodeReader &Reader) {
  clift::ValueType PointeeType;
  if (Reader.readType(PointeeType).failed())
    return {};

  uint64_t PointerSize;
  if (Reader.readVarInt(PointerSize).failed())
    return {};

  bool Const;
  if (readBool(Const, Reader).failed())
    return {};

  return clift::PointerType::get(PointeeType, PointerSize, Const);
}

static void writeType(clift::PointerType Type,
                      mlir::DialectBytecodeWriter &Writer) {
  Writer.writeType(Type.getPointeeType());
  Writer.writeVarInt(Type.getPointerSize());
  writeBool(Type.getIsConst(), Writer);
}

//===------------------------------ ArrayType -----------------------------===//

mlir::LogicalResult ArrayType::verify(EmitErrorType EmitError,
                                      ValueType ElementType,
                                      uint64_t ElementCount) {
  if (not isObjectType(ElementType))
    return EmitError() << "Array type element type must be an object type.";
  if (ElementCount == 0)
    return EmitError() << "Array type must have no less than one element.";

  return mlir::success();
}

ValueType ArrayType::addConst() const {
  auto ElementT = getElementType();
  auto NewElementT = ElementT.addConst();

  if (ElementT == NewElementT)
    return *this;

  return get(getContext(), NewElementT, getElementsCount());
}

ValueType ArrayType::removeConst() const {
  auto ElementT = getElementType();
  auto NewElementT = ElementT.removeConst();

  if (ElementT == NewElementT)
    return *this;

  return get(getContext(), NewElementT, getElementsCount());
}

template<std::same_as<clift::ArrayType>>
static clift::ArrayType readType(mlir::DialectBytecodeReader &Reader) {
  clift::ValueType ElementType;
  if (Reader.readType(ElementType).failed())
    return {};

  uint64_t ElementCount;
  if (Reader.readVarInt(ElementCount).failed())
    return {};

  return clift::ArrayType::get(ElementType, ElementCount);
}

static void writeType(clift::ArrayType Type,
                      mlir::DialectBytecodeWriter &Writer) {
  Writer.writeType(Type.getElementType());
  Writer.writeVarInt(Type.getElementsCount());
}

//===------------------------------ EnumType ------------------------------===//

bool EnumType::getAlias(llvm::raw_ostream &OS) const {
  if (getIsConst())
    return false;

  return getDefinedTypeAlias(*this, OS);
}

clift::ValueType EnumType::addConst() const {
  return Base::get(getContext(), getDefinition(), /*IsConst=*/true);
}

clift::ValueType EnumType::removeConst() const {
  return Base::get(getContext(), getDefinition(), /*IsConst=*/false);
}

mlir::Type EnumType::parse(mlir::AsmParser &Parser) {
  mlir::SMLoc Loc = Parser.getCurrentLocation();

  if (Parser.parseLess().failed())
    return {};

  std::string Handle;
  if (Parser.parseString(&Handle).failed())
    return {};

  std::string Name;
  if (mlir::parseCliftDebugName(Parser, Name).failed())
    return {};

  if (Parser.parseColon().failed())
    return {};

  clift::ValueType UnderlyingType;
  if (Parser.parseType(UnderlyingType).failed())
    return {};

  llvm::SmallVector<clift::EnumFieldAttr> Enumerators;
  auto ParseEnumerator = [&]() -> mlir::ParseResult {
    std::string Handle;
    if (Parser.parseString(&Handle).failed())
      return mlir::failure();

    std::string Name;
    if (mlir::parseCliftDebugName(Parser, Name).failed())
      return mlir::failure();

    if (Parser.parseColon().failed())
      return mlir::failure();

    uint64_t Value;
    if (Parser.parseInteger(Value).failed())
      return mlir::failure();

    auto NameAttr = makeNameAttr<clift::EnumFieldAttr>(Parser.getContext(),
                                                       Handle,
                                                       Name);

    auto Attr = clift::EnumFieldAttr::getChecked(getEmitError(Parser, Loc),
                                                 Parser.getContext(),
                                                 llvm::StringRef(Handle),
                                                 NameAttr,
                                                 Value);

    if (not Attr)
      return mlir::failure();

    Enumerators.push_back(Attr);
    return mlir::success();
  };

  if (Parser
        .parseCommaSeparatedList(mlir::AsmParser::Delimiter::Braces,
                                 ParseEnumerator,
                                 " in enumerator list")
        .failed())
    return {};

  if (Parser.parseGreater().failed())
    return {};

  auto NameAttr = makeNameAttr<EnumAttr>(Parser.getContext(), Handle, Name);
  auto Attr = EnumAttr::getChecked(getEmitError(Parser, Loc),
                                   Parser.getContext(),
                                   llvm::StringRef(Handle),
                                   NameAttr,
                                   UnderlyingType,
                                   llvm::ArrayRef(Enumerators));

  if (not Attr)
    return {};

  return EnumType::get(Attr);
}

void EnumType::print(mlir::AsmPrinter &Printer) const {
  Printer << "<";
  printString(Printer, getHandle());
  mlir::printCliftDebugName(Printer, getName());

  Printer << " : ";
  Printer.printType(getUnderlyingType());

  Printer << " {";
  if (not getFields().empty()) {
    for (auto [I, E] : llvm::enumerate(getFields())) {
      if (I != 0)
        Printer << ',';

      Printer << "\n  ";
      printString(Printer, E.getHandle());
      mlir::printCliftDebugName(Printer, E.getName());
      Printer << " : " << E.getRawValue();
    }
    Printer << '\n';
  }
  Printer << "}>";
}

template<std::same_as<clift::EnumType>>
static clift::EnumType readType(mlir::DialectBytecodeReader &Reader) {
  clift::EnumAttr Definition;
  if (Reader.readAttribute(Definition).failed())
    return {};

  bool Const;
  if (readBool(Const, Reader).failed())
    return {};

  return clift::EnumType::get(Reader.getContext(), Definition, Const);
}

static void writeType(clift::EnumType Type,
                      mlir::DialectBytecodeWriter &Writer) {
  Writer.writeAttribute(Type.getDefinition());
  writeBool(Type.getIsConst(), Writer);
}

//===----------------------------- TypedefType ----------------------------===//

bool TypedefType::getAlias(llvm::raw_ostream &OS) const {
  if (getIsConst())
    return false;

  return getDefinedTypeAlias(*this, OS);
}

uint64_t TypedefType::getByteSize() const {
  return getUnderlyingType().getByteSize();
}

clift::ValueType TypedefType::addConst() const {
  return Base::get(getContext(), getDefinition(), /*IsConst=*/true);
}

clift::ValueType TypedefType::removeConst() const {
  return Base::get(getContext(), getDefinition(), /*IsConst=*/false);
}

mlir::Type TypedefType::parse(mlir::AsmParser &Parser) {
  mlir::SMLoc Loc = Parser.getCurrentLocation();

  if (Parser.parseLess().failed())
    return {};

  std::string Handle;
  if (Parser.parseString(&Handle).failed())
    return {};

  std::string Name;
  if (Parser.parseOptionalKeyword("as").succeeded()) {
    if (Parser.parseString(&Name).failed())
      return {};
  }

  if (Parser.parseColon().failed())
    return {};

  clift::ValueType UnderlyingType;
  if (Parser.parseType(UnderlyingType).failed())
    return {};

  if (Parser.parseGreater().failed())
    return {};

  auto NameAttr = makeNameAttr<TypedefAttr>(Parser.getContext(), Handle, Name);
  auto Attr = TypedefAttr::getChecked(getEmitError(Parser, Loc),
                                      Parser.getContext(),
                                      llvm::StringRef(Handle),
                                      NameAttr,
                                      UnderlyingType);

  return TypedefType::get(Attr);
}

void TypedefType::print(mlir::AsmPrinter &Printer) const {
  Printer << "<";
  printString(Printer, getHandle());
  mlir::printCliftDebugName(Printer, getName());

  Printer << " : ";
  Printer.printType(getUnderlyingType());
  Printer << ">";
}

template<std::same_as<clift::TypedefType>>
static clift::TypedefType readType(mlir::DialectBytecodeReader &Reader) {
  clift::TypedefAttr Definition;
  if (Reader.readAttribute(Definition).failed())
    return {};

  bool Const;
  if (readBool(Const, Reader).failed())
    return {};

  return clift::TypedefType::get(Reader.getContext(), Definition, Const);
}

static void writeType(clift::TypedefType Type,
                      mlir::DialectBytecodeWriter &Writer) {
  Writer.writeAttribute(Type.getDefinition());
  writeBool(Type.getIsConst(), Writer);
}

//===---------------------------- FunctionType ----------------------------===//

mlir::LogicalResult FunctionType::verify(EmitErrorType EmitError,
                                         llvm::StringRef Handle,
                                         MutableStringAttr Name,
                                         mlir::Type ReturnType,
                                         llvm::ArrayRef<mlir::Type> Args) {
  auto R = mlir::dyn_cast<clift::ValueType>(ReturnType);
  if (not R)
    return EmitError() << "Function return type must be a ValueType";

  for (mlir::Type ArgumentType : Args) {
    auto T = mlir::dyn_cast<clift::ValueType>(ArgumentType);
    if (not T)
      return EmitError() << "Function argument types must be ValueTypes";

    if (not isObjectType(T))
      return EmitError() << "Function parameter type must be an object type";
    if (isArrayType(T))
      return EmitError() << "Function parameter type may not be an array type";
  }

  auto EmitReturnError = [&]() -> mlir::InFlightDiagnostic {
    return EmitError() << "Function return type ";
  };

  if (verifyReturnType(EmitReturnError, R).failed())
    return mlir::failure();

  return mlir::success();
}

bool FunctionType::getAlias(llvm::raw_ostream &OS) const {
  return getDefinedTypeAlias(*this, OS);
}

llvm::ArrayRef<mlir::Type> FunctionType::getResultTypes() const {
  return ArrayRef<Type>(getImpl()->return_type);
}

mlir::Type FunctionType::parse(mlir::AsmParser &Parser) {
  mlir::SMLoc Loc = Parser.getCurrentLocation();

  if (Parser.parseLess().failed())
    return {};

  std::string Handle;
  if (Parser.parseString(&Handle).failed())
    return {};

  std::string Name;
  if (mlir::parseCliftDebugName(Parser, Name).failed())
    return {};

  if (Parser.parseColon().failed())
    return {};

  clift::ValueType ReturnType;
  if (Parser.parseType(ReturnType).failed())
    return {};

  llvm::SmallVector<clift::ValueType> ParameterTypes;
  const auto ParseParameterType = [&Parser,
                                   &ParameterTypes]() -> mlir::ParseResult {
    clift::ValueType ParameterType;
    if (Parser.parseType(ParameterType))
      return mlir::failure();

    ParameterTypes.push_back(ParameterType);
    return mlir::success();
  };

  if (Parser
        .parseCommaSeparatedList(mlir::AsmParser::Delimiter::Paren,
                                 ParseParameterType,
                                 " in parameter list")
        .failed())
    return {};

  if (Parser.parseGreater().failed())
    return {};

  auto NameAttr = makeNameAttr<FunctionType>(Parser.getContext(), Handle, Name);
  return FunctionType::getChecked(getEmitError(Parser, Loc),
                                  Parser.getContext(),
                                  llvm::StringRef(Handle),
                                  NameAttr,
                                  ReturnType,
                                  llvm::ArrayRef(ParameterTypes));
}

void FunctionType::print(mlir::AsmPrinter &Printer) const {
  Printer << "<";
  printString(Printer, getHandle());
  mlir::printCliftDebugName(Printer, getName());

  Printer << " : ";
  Printer.printType(getReturnType());
  Printer << "(";

  bool Comma = false;
  for (mlir::Type ParameterType : getArgumentTypes()) {
    if (Comma)
      Printer << ", ";
    Comma = true;

    Printer.printType(ParameterType);
  }

  Printer << ")";
  Printer << ">";
}

template<std::same_as<clift::FunctionType>>
static clift::FunctionType readType(mlir::DialectBytecodeReader &Reader) {
  auto ReadType = [&](mlir::Type &Type) -> mlir::LogicalResult {
    clift::ValueType ValueType;
    if (Reader.readType(ValueType).failed())
      return mlir::failure();
    Type = ValueType;
    return mlir::success();
  };

  llvm::StringRef Handle;
  if (Reader.readString(Handle).failed())
    return {};

  llvm::StringRef Name;
  if (Reader.readString(Name).failed())
    return {};

  mlir::Type ReturnType;
  if (ReadType(ReturnType).failed())
    return {};

  llvm::SmallVector<mlir::Type> ParameterTypes;
  if (Reader.readList(ParameterTypes, ReadType).failed())
    return {};

  auto NameAttr = makeNameAttr<FunctionType>(Reader.getContext(), Handle, Name);
  return clift::FunctionType::getChecked(getEmitError(Reader),
                                         Reader.getContext(),
                                         Handle,
                                         NameAttr,
                                         ReturnType,
                                         std::move(ParameterTypes));
}

static void writeType(clift::FunctionType Type,
                      mlir::DialectBytecodeWriter &Writer) {
  Writer.writeOwnedString(Type.getHandle());
  Writer.writeOwnedString(Type.getName());
  Writer.writeType(Type.getReturnType());
  Writer.writeList(Type.getArgumentTypes(),
                   [&](mlir::Type Type) { Writer.writeType(Type); });
}

//===----------------------------- Class types ----------------------------===//

// During parsing, contains (as an opaque pointer) the class type definition
// attribute (StructAttr or UnionAttr) of each class currently being parsed.
//
// During printing, contains (as an opaque pointer) each class type (StructType
// or UnionType) currently being printed.
static thread_local llvm::SmallPtrSet<const void *, 8> ClassRecursionSet;

template<typename TypeT, typename AttrT>
static TypeT parseClassType(mlir::AsmParser &Parser) {
  constexpr bool IsStruct = std::is_same_v<TypeT, StructType>;

  mlir::SMLoc Loc = Parser.getCurrentLocation();

  if (Parser.parseLess().failed())
    return {};

  std::string Handle;
  if (Parser.parseString(&Handle).failed())
    return {};

  auto IncompleteAttr = AttrT::get(Parser.getContext(), Handle);
  const void *Opaque = IncompleteAttr.getAsOpaquePointer();

  auto [Iterator, Inserted] = ClassRecursionSet.insert(Opaque);
  auto Guard = llvm::make_scope_exit([&]() {
    if (Inserted)
      ClassRecursionSet.erase(Opaque);
  });

  if (Parser.parseOptionalGreater().succeeded())
    return TypeT::get(Parser.getContext(), Handle);

  std::string Name;
  if (mlir::parseCliftDebugName(Parser, Name).failed())
    return {};

  if (Parser.parseColon().failed())
    return {};

  uint64_t Size;
  if constexpr (IsStruct) {
    if (Parser.parseKeyword("size").failed())
      return {};

    if (Parser.parseLParen().failed())
      return {};

    if (Parser.parseInteger(Size).failed())
      return {};

    if (Parser.parseRParen().failed())
      return {};
  }

  llvm::SmallVector<FieldAttr> Fields;
  auto ParseField = [&]() -> mlir::ParseResult {
    auto FieldLoc = Parser.getCurrentLocation();

    std::string Handle;
    if (Parser.parseString(&Handle).failed())
      return mlir::failure();

    std::string Name;
    if (mlir::parseCliftDebugName(Parser, Name))
      return mlir::failure();

    if (Parser.parseColon().failed())
      return mlir::failure();

    uint64_t Offset = 0;
    if constexpr (IsStruct) {
      if (Parser.parseKeyword("offset").failed())
        return mlir::failure();

      if (Parser.parseLParen().failed())
        return mlir::failure();

      if (Parser.parseInteger(Offset).failed())
        return mlir::failure();

      if (Parser.parseRParen().failed())
        return mlir::failure();
    }

    clift::ValueType Type;
    if (Parser.parseType(Type).failed())
      return mlir::failure();

    auto NameAttr = makeNameAttr<FieldAttr>(Parser.getContext(), Handle, Name);
    auto Attr = FieldAttr::getChecked(getEmitError(Parser, FieldLoc),
                                      Parser.getContext(),
                                      llvm::StringRef(Handle),
                                      NameAttr,
                                      Offset,
                                      Type);

    if (not Attr)
      return mlir::failure();

    Fields.push_back(Attr);
    return mlir::success();
  };

  if (Parser
        .parseCommaSeparatedList(mlir::AsmParser::Delimiter::Braces,
                                 ParseField,
                                 " in field list")
        .failed())
    return {};

  if (Parser.parseGreater().failed())
    return {};

  auto GetCompleteType = [&](const auto &...Args) -> TypeT {
    auto NameAttr = makeNameAttr<AttrT>(Parser.getContext(), Handle, Name);
    auto Attr = AttrT::getChecked(getEmitError(Parser, Loc),
                                  Parser.getContext(),
                                  llvm::StringRef(Handle),
                                  NameAttr,
                                  Args...,
                                  llvm::ArrayRef(Fields));

    if (not Attr)
      return {};

    return TypeT::get(Attr);
  };

  if constexpr (IsStruct) {
    return GetCompleteType(Size);
  } else {
    return GetCompleteType();
  }
}

template<typename TypeT>
static void printClassType(TypeT Type, mlir::AsmPrinter &Printer) {
  constexpr bool IsStruct = std::is_same_v<TypeT, StructType>;

  auto PrintIncomplete = [&]() {
    Printer << '<';
    printString(Printer, Type.getHandle());
    Printer << '>';
  };

  const void *Opaque = Type.getAsOpaquePointer();
  const void *CurrentAlias = Printer.getCurrentTypeOrAttrAlias();

  // When printing an alias (CurrentAlias != nullptr), the full definition is
  // only printed if that alias is for this exact type (CurrentAlias == Opaque).
  if ((CurrentAlias != nullptr and CurrentAlias != Opaque))
    return PrintIncomplete();

  // Whether printing an alias or not, the full definition is never printed
  // recursively within the definition of the same type.
  if (not ClassRecursionSet.insert(Opaque).second)
    return PrintIncomplete();

  auto Guard = llvm::make_scope_exit([&]() {
    ClassRecursionSet.erase(Opaque);
  });

  Printer << "<";
  printString(Printer, Type.getHandle());
  mlir::printCliftDebugName(Printer, Type.getName());

  Printer << " : ";
  if constexpr (IsStruct) {
    Printer << "size(" << Type.getSize() << ") ";
  }

  Printer << "{";
  if (not Type.getFields().empty()) {
    for (auto [I, S] : llvm::enumerate(Type.getFields())) {
      if (I != 0)
        Printer << ',';

      Printer << "\n  ";
      printString(Printer, S.getHandle());
      mlir::printCliftDebugName(Printer, S.getName());

      Printer << " :";

      if constexpr (IsStruct) {
        Printer << " offset(" << S.getOffset() << ")";
      }

      Printer << ' ' << S.getType();
    }
    Printer << '\n';
  }
  Printer << "}>";
}

template<typename AttrT>
static uint64_t getClassTypeSize(AttrT Definition) {
  if (not Definition.hasDefinition()) {
    if (ClassRecursionSet.contains(Definition.getAsOpaquePointer())) {
      // We can return zero for any struct still being parsed. This avoids
      // assert failure when verifying recursively defined structs. Recursively
      // defined class types are rejected later during module verification.
      return 0;
    }
  }

  return Definition.getSize();
}

static thread_local llvm::SmallPtrSet<const void *, 32> BytecodeClassTypeSet;

template<typename TypeT, typename AttrT>
static TypeT readClassType(mlir::DialectBytecodeReader &Reader) {
  constexpr bool IsStruct = std::is_same_v<TypeT, StructType>;

  llvm::StringRef Handle;
  if (Reader.readString(Handle).failed())
    return {};

  bool Const;
  if (readBool(Const, Reader).failed())
    return {};

  auto Type = TypeT::get(Reader.getContext(), Handle);
  const void *Opaque = Type.getAsOpaquePointer();
  auto [Iterator, Inserted] = BytecodeClassTypeSet.insert(Opaque);

  auto Guard = llvm::make_scope_exit([&]() {
    if (Inserted)
      BytecodeClassTypeSet.erase(Opaque);
  });

  // This is the complicated part of class type serialization.
  //
  // First, a little bit of background:
  //
  // The bytecode serializer first visits each type recursively, recording all
  // reachable types and assigning them numeric ids. After that it visits each
  // type again non-recursively, serializing them into a table, with references
  // to other types serialized using their numeric ids. The same applies to
  // attributes
  //
  // The bytecode deserializer deserializes types and attributes from the table
  // on demand. When a type or attribute is fully deserialized for the first
  // time, it is remembered and not deserialized again.
  //
  // Now the problem with recursive types:
  //
  // When deserializing a recursive class, using the normal rules, it is never
  // *fully* deserialized, because the same type must be deserialized again and
  // again forever. To get around this we employ a small hack:
  //
  // 1. Instead of storing the type in a single blob as usual, we store the
  //    handle and constness of type, as well as a BytecodeClassAttr attribute
  //    referencing the type itself. This creates an indirection in the table
  //    as explained above. The definition of the class type is deserialized
  //    as part of the attribute deserialization.
  //    See `readClassDefinition` and `writeClassDefinition` below.
  //
  // 2. When recursion is detected during deserialization, instead of
  //    deserializing the contained attribute (which is resolved through the
  //    table) again, we read the underlying integer and simply skip it, ending
  //    the recursion. At this point the class type has already been defined.

  if (Inserted) {
    clift::BytecodeClassAttr HelperAttr;
    if (Reader.readAttribute(HelperAttr).failed())
      return {};
    revng_assert(HelperAttr.getType() == Type);
    revng_assert(Type.isComplete());
  } else {
    // Here we skip the deserialization of the helper attribute by reading and
    // ignoring the underlying table reference:
    uint64_t HelperAttrReference;
    if (Reader.readVarInt(HelperAttrReference).failed())
      return {};
  }

  if (Const)
    Type = mlir::cast<TypeT>(Type.addConst());

  return Type;
}

template<typename TypeT>
static void writeClassType(TypeT Type, mlir::DialectBytecodeWriter &Writer) {
  constexpr bool IsStruct = std::is_same_v<TypeT, StructType>;

  Writer.writeOwnedString(Type.getHandle());
  writeBool(Type.getIsConst(), Writer);
  Writer.writeAttribute(BytecodeClassAttr::get(Type.getContext(), Type));
}

template<typename TypeT, typename AttrT>
static TypeT readClassDefinition(mlir::DialectBytecodeReader &Reader) {
  constexpr bool IsStruct = std::is_same_v<TypeT, StructType>;

  llvm::StringRef Handle;
  if (Reader.readString(Handle).failed())
    return {};

  llvm::StringRef Name;
  if (Reader.readString(Name).failed())
    return {};

  uint64_t Size = 0;
  if constexpr (IsStruct) {
    if (Reader.readVarInt(Size).failed())
      return {};
  }

  auto ReadField = [&](clift::FieldAttr &Field) -> mlir::LogicalResult {
    llvm::StringRef Handle;
    if (Reader.readString(Handle).failed())
      return mlir::failure();

    llvm::StringRef Name;
    if (Reader.readString(Name).failed())
      return mlir::failure();

    uint64_t Offset = 0;
    if constexpr (IsStruct) {
      if (Reader.readVarInt(Offset).failed())
        return mlir::failure();
    }

    clift::ValueType Type;
    if (Reader.readType(Type).failed())
      return mlir::failure();

    auto NameAttr = makeNameAttr<clift::FieldAttr>(Reader.getContext(),
                                                   Handle,
                                                   Name);

    Field = clift::FieldAttr::getChecked(getEmitError(Reader),
                                         Reader.getContext(),
                                         llvm::StringRef(Handle),
                                         NameAttr,
                                         Offset,
                                         Type);

    return mlir::success(static_cast<bool>(Field));
  };

  llvm::SmallVector<clift::FieldAttr> Fields;
  if (Reader.readList(Fields, ReadField).failed())
    return {};

  auto GetCompleteType = [&](const auto &...Args) -> TypeT {
    auto NameAttr = makeNameAttr<AttrT>(Reader.getContext(), Handle, Name);
    auto Attr = AttrT::getChecked(getEmitError(Reader),
                                  Reader.getContext(),
                                  llvm::StringRef(Handle),
                                  NameAttr,
                                  Args...,
                                  llvm::ArrayRef(Fields));

    if (not Attr)
      return {};

    return TypeT::get(Attr);
  };

  if constexpr (IsStruct) {
    return GetCompleteType(Size);
  } else {
    return GetCompleteType();
  }
}

template<typename TypeT>
static void
writeClassDefinition(TypeT Type, mlir::DialectBytecodeWriter &Writer) {
  constexpr bool IsStruct = std::is_same_v<TypeT, StructType>;

  Writer.writeOwnedString(Type.getHandle());
  Writer.writeOwnedString(Type.getName());

  if constexpr (IsStruct) {
    Writer.writeVarInt(Type.getSize());
  }

  Writer.writeList(Type.getFields(), [&](clift::FieldAttr Field) {
    Writer.writeOwnedString(Field.getHandle());
    Writer.writeOwnedString(Field.getName());

    if constexpr (IsStruct) {
      Writer.writeVarInt(Field.getOffset());
    }

    Writer.writeType(Field.getType());
  });
}

//===----------------------------- StructType -----------------------------===//

uint64_t StructType::getSize() const {
  return getClassTypeSize(getDefinition());
}

bool StructType::getAlias(llvm::raw_ostream &OS) const {
  if (getIsConst())
    return false;

  return getDefinedTypeAlias(*this, OS);
}

mlir::Type StructType::parse(mlir::AsmParser &Parser) {
  return parseClassType<StructType, StructAttr>(Parser);
}

void StructType::print(mlir::AsmPrinter &Printer) const {
  return printClassType(*this, Printer);
}

template<std::same_as<clift::StructType>>
static clift::StructType readType(mlir::DialectBytecodeReader &Reader) {
  return readClassType<StructType, StructAttr>(Reader);
}

static void writeType(clift::StructType Type,
                      mlir::DialectBytecodeWriter &Writer) {
  return writeClassType<StructType>(Type, Writer);
}

clift::StructType
clift::readStructDefinition(mlir::DialectBytecodeReader &Reader) {
  return readClassDefinition<StructType, StructAttr>(Reader);
}

void clift::writeStructDefinition(clift::StructType Type,
                                  mlir::DialectBytecodeWriter &Writer) {
  return writeClassDefinition(Type, Writer);
}

//===------------------------------ UnionType -----------------------------===//

uint64_t UnionType::getSize() const {
  return getClassTypeSize(getDefinition());
}

bool UnionType::getAlias(llvm::raw_ostream &OS) const {
  if (getIsConst())
    return false;

  return getDefinedTypeAlias(*this, OS);
}

mlir::Type UnionType::parse(mlir::AsmParser &Parser) {
  return parseClassType<UnionType, UnionAttr>(Parser);
}

void UnionType::print(mlir::AsmPrinter &Printer) const {
  return printClassType(*this, Printer);
}

template<std::same_as<clift::UnionType>>
static clift::UnionType readType(mlir::DialectBytecodeReader &Reader) {
  return readClassType<UnionType, UnionAttr>(Reader);
}

static void writeType(clift::UnionType Type,
                      mlir::DialectBytecodeWriter &Writer) {
  return writeClassType<UnionType>(Type, Writer);
}

clift::UnionType
clift::readUnionDefinition(mlir::DialectBytecodeReader &Reader) {
  return readClassDefinition<UnionType, UnionAttr>(Reader);
}

void clift::writeUnionDefinition(clift::UnionType Type,
                                 mlir::DialectBytecodeWriter &Writer) {
  return writeClassDefinition(Type, Writer);
}

//===---------------------------- Type helpers ----------------------------===//

TypedefDecomposition clift::decomposeTypedef(ValueType Type) {
  bool HasConstTypedef = false;

  while (true) {
    auto T = mlir::dyn_cast<TypedefType>(Type);
    if (not T)
      break;

    Type = T.getUnderlyingType();
    HasConstTypedef |= T.isConst();
  }

  return { Type, HasConstTypedef };
}

ValueType clift::dealias(ValueType Type, bool IgnoreQualifiers) {
  auto &&[UnderlyingType, HasConstTypedef] = decomposeTypedef(Type);

  if (HasConstTypedef and not IgnoreQualifiers)
    UnderlyingType = UnderlyingType.addConst();

  return UnderlyingType;
}

mlir::Type clift::removeConst(mlir::Type Type) {
  if (auto ValueT = mlir::dyn_cast<clift::ValueType>(Type))
    Type = ValueT.removeConst();

  return Type;
}

bool clift::equivalent(mlir::Type Lhs, mlir::Type Rhs) {
  if (Lhs == Rhs)
    return true;

  if (not Lhs or not Rhs)
    return false;

  auto LhsVT = mlir::dyn_cast<clift::ValueType>(Lhs);
  if (not LhsVT)
    return false;

  auto RhsVT = mlir::dyn_cast<clift::ValueType>(Rhs);
  if (not RhsVT)
    return false;

  return LhsVT.removeConst() == RhsVT.removeConst();
}

bool clift::isModifiableType(ValueType Type) {
  auto &&[UnderlyingType, HasConst] = decomposeTypedef(Type);
  return not HasConst and not UnderlyingType.isConst();
}

bool clift::isIntegerKind(PrimitiveKind Kind) {
  switch (Kind) {
  case PrimitiveKind::GenericKind:
  case PrimitiveKind::PointerOrNumberKind:
  case PrimitiveKind::NumberKind:
  case PrimitiveKind::UnsignedKind:
  case PrimitiveKind::SignedKind:
    return true;

  case PrimitiveKind::VoidKind:
  case PrimitiveKind::FloatKind:
    break;
  }
  return false;
}

PrimitiveType clift::getUnderlyingIntegerType(ValueType Type) {
  Type = dealias(Type, /*IgnoreQualifiers=*/true);

  if (auto T = mlir::dyn_cast<PrimitiveType>(Type))
    return isIntegerKind(T.getKind()) ? T : nullptr;

  if (auto T = mlir::dyn_cast<EnumType>(Type)) {
    return mlir::cast<PrimitiveType>(dealias(T.getUnderlyingType()));
  }

  return nullptr;
}

bool clift::isCompleteType(ValueType Type) {
  Type = dealias(Type, /*IgnoreQualifiers=*/true);

  if (auto D = mlir::dyn_cast<StructType>(Type))
    return D.isComplete();

  if (auto D = mlir::dyn_cast<UnionType>(Type))
    return D.isComplete();

  if (auto T = mlir::dyn_cast<ArrayType>(Type))
    return isCompleteType(T.getElementType());

  return true;
}

bool clift::isVoid(ValueType Type) {
  Type = dealias(Type, /*IgnoreQualifiers=*/true);

  if (auto T = mlir::dyn_cast<PrimitiveType>(Type))
    return T.getKind() == PrimitiveKind::VoidKind;

  return false;
}

bool clift::isScalarType(ValueType Type) {
  Type = dealias(Type, /*IgnoreQualifiers=*/true);

  if (auto T = mlir::dyn_cast<PrimitiveType>(Type))
    return T.getKind() != PrimitiveKind::VoidKind;

  return mlir::isa<EnumType, PointerType>(Type);
}

bool clift::isPrimitiveIntegerType(ValueType Type) {
  Type = dealias(Type, /*IgnoreQualifiers=*/true);

  if (auto T = mlir::dyn_cast<PrimitiveType>(Type))
    return isIntegerKind(T.getKind());

  return false;
}

bool clift::isIntegerType(ValueType Type) {
  Type = dealias(Type, /*IgnoreQualifiers=*/true);

  if (auto T = mlir::dyn_cast<PrimitiveType>(Type))
    return isIntegerKind(T.getKind());

  return mlir::isa<EnumType>(Type);
}

bool clift::isFloatType(ValueType Type) {
  Type = dealias(Type, /*IgnoreQualifiers=*/true);

  if (auto T = mlir::dyn_cast<PrimitiveType>(Type))
    return T.getKind() == PrimitiveKind::FloatKind;

  return false;
}

PointerType clift::getPointerType(ValueType Type) {
  return mlir::dyn_cast<PointerType>(dealias(Type, /*IgnoreQualifiers=*/true));
}

bool clift::isPointerType(ValueType Type) {
  return static_cast<bool>(getPointerType(Type));
}

bool clift::isObjectType(ValueType Type) {
  Type = dealias(Type, /*IgnoreQualifiers=*/true);

  if (auto T = mlir::dyn_cast<PrimitiveType>(Type)) {
    if (T.getKind() == PrimitiveKind::VoidKind)
      return false;
  }

  return not mlir::isa<clift::FunctionType>(Type);
}

bool clift::isArrayType(ValueType Type) {
  return mlir::isa<ArrayType>(dealias(Type, /*IgnoreQualifiers=*/true));
}

bool clift::isEnumType(ValueType Type) {
  return mlir::isa<EnumType>(dealias(Type, /*IgnoreQualifiers=*/true));
}

bool clift::isClassType(ValueType Type) {
  return mlir::isa<ClassType>(dealias(Type, /*IgnoreQualifiers=*/true));
}

bool clift::isFunctionType(ValueType Type) {
  Type = dealias(Type, /*IgnoreQualifiers=*/true);
  return mlir::isa<clift::FunctionType>(Type);
}

clift::FunctionType
clift::getFunctionOrFunctionPointerFunctionType(ValueType Type) {
  Type = dealias(Type, /*IgnoreQualifiers=*/true);

  if (auto P = mlir::dyn_cast<PointerType>(Type))
    Type = dealias(P.getPointeeType(), /*IgnoreQualifiers=*/true);

  return mlir::dyn_cast<clift::FunctionType>(Type);
}

mlir::LogicalResult clift::verifyReturnType(EmitErrorType EmitError,
                                            ValueType ReturnType) {
  ValueType UnderlyingType = dealias(ReturnType, /*IgnoreQualifiers=*/true);

  if (isVoid(UnderlyingType))
    return mlir::success();

  if (isObjectType(UnderlyingType)) {
    if (isArrayType(UnderlyingType))
      return EmitError() << "cannot be an array type.";

    // Check for pointers to array or function types. Function types returning
    // pointers to arrays or functions cannot be represented in C.
    {
      ValueType T = ReturnType;
      while (auto P = mlir::dyn_cast<PointerType>(T))
        T = P.getPointeeType();

      if (mlir::isa<ArrayType, FunctionType>(T))
        return EmitError() << "cannot be pointer to array or function type.";
    }

    return mlir::success();
  }

  return EmitError() << "must be an object type or void.";
}

//===---------------------------- CliftDialect ----------------------------===//

void CliftDialect::registerTypes() {
  addTypes</* Include the auto-generated clift types */
#define GET_TYPEDEF_LIST
#include "revng/mlir/Dialect/Clift/IR/CliftOpsTypes.cpp.inc"
           /* End of auto-generated list */>();
}

static clift::ValueType parseConstType(mlir::DialectAsmParser &Parser) {
  if (Parser.parseLess().failed())
    return {};

  clift::ValueType UnderlyingType;
  if (Parser.parseType(UnderlyingType).failed())
    return {};

  if (Parser.parseGreater().failed())
    return {};

  return UnderlyingType.addConst();
}

static void printConstType(clift::ValueType Type,
                           mlir::DialectAsmPrinter &Printer) {
  Printer << "const<";
  Printer.printType(Type);
  Printer << ">";
}

/// Parse a type registered to this dialect
mlir::Type CliftDialect::parseType(mlir::DialectAsmParser &Parser) const {
  const llvm::SMLoc TypeLoc = Parser.getCurrentLocation();

  llvm::StringRef Mnemonic;
  if (mlir::Type T; generatedTypeParser(Parser, &Mnemonic, T).has_value())
    return T;

  if (Mnemonic == "const")
    return parseConstType(Parser);

  Parser.emitError(TypeLoc) << "unknown type `" << Mnemonic << "` in dialect `"
                            << getNamespace() << "`";
  return {};
}

/// Print a type registered to this dialect
void CliftDialect::printType(mlir::Type Type,
                             mlir::DialectAsmPrinter &Printer) const {
  if (auto ConstType = mlir::dyn_cast<clift::ValueType>(Type)) {
    if (ConstType.isConst())
      return printConstType(ConstType.removeConst(), Printer);
  }

  if (mlir::succeeded(generatedTypePrinter(Type, Printer)))
    return;

  revng_abort("cannot print type");
}

namespace {

enum class CliftTypeKind : uint8_t {
  Label,
  Primitive,
  Pointer,
  Array,
  Enum,
  Typedef,
  Function,
  Struct,
  Union,

  N
};

} // namespace

static mlir::LogicalResult readTypeKind(CliftTypeKind &TypeKind,
                                        mlir::DialectBytecodeReader &Reader) {
  uint64_t Value;
  if (Reader.readVarInt(Value).failed())
    return mlir::failure();

  if (Value >= static_cast<uint64_t>(CliftTypeKind::N))
    return mlir::failure();

  TypeKind = static_cast<CliftTypeKind>(Value);
  return mlir::success();
}

mlir::Type clift::readType(mlir::DialectBytecodeReader &Reader) {
  CliftTypeKind TypeKind;
  if (readTypeKind(TypeKind, Reader).failed())
    return {};

  switch (TypeKind) {
  case CliftTypeKind::Label:
    return ::readType<clift::LabelType>(Reader);
  case CliftTypeKind::Primitive:
    return ::readType<clift::PrimitiveType>(Reader);
  case CliftTypeKind::Pointer:
    return ::readType<clift::PointerType>(Reader);
  case CliftTypeKind::Array:
    return ::readType<clift::ArrayType>(Reader);
  case CliftTypeKind::Enum:
    return ::readType<clift::EnumType>(Reader);
  case CliftTypeKind::Typedef:
    return ::readType<clift::TypedefType>(Reader);
  case CliftTypeKind::Function:
    return ::readType<clift::FunctionType>(Reader);
  case CliftTypeKind::Struct:
    return ::readType<clift::StructType>(Reader);
  case CliftTypeKind::Union:
    return ::readType<clift::UnionType>(Reader);
  case CliftTypeKind::N:
    break;
  }
  revng_abort();
}

mlir::LogicalResult clift::writeType(mlir::Type Type,
                                     mlir::DialectBytecodeWriter &Writer) {
  auto Write = [&](auto T, CliftTypeKind TypeKind) {
    Writer.writeVarInt(static_cast<uint64_t>(TypeKind));
    ::writeType(T, Writer);
    return mlir::success();
  };

  if (auto T = mlir::dyn_cast<clift::LabelType>(Type))
    return Write(T, CliftTypeKind::Label);
  if (auto T = mlir::dyn_cast<clift::PrimitiveType>(Type))
    return Write(T, CliftTypeKind::Primitive);
  if (auto T = mlir::dyn_cast<clift::PointerType>(Type))
    return Write(T, CliftTypeKind::Pointer);
  if (auto T = mlir::dyn_cast<clift::ArrayType>(Type))
    return Write(T, CliftTypeKind::Array);
  if (auto T = mlir::dyn_cast<clift::EnumType>(Type))
    return Write(T, CliftTypeKind::Enum);
  if (auto T = mlir::dyn_cast<clift::TypedefType>(Type))
    return Write(T, CliftTypeKind::Typedef);
  if (auto T = mlir::dyn_cast<clift::FunctionType>(Type))
    return Write(T, CliftTypeKind::Function);
  if (auto T = mlir::dyn_cast<clift::StructType>(Type))
    return Write(T, CliftTypeKind::Struct);
  if (auto T = mlir::dyn_cast<clift::UnionType>(Type))
    return Write(T, CliftTypeKind::Union);

  return mlir::failure();
}
