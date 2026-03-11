//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <optional>
#include <string>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/StringExtras.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Support/LogicalResult.h"

#include "revng/Clift/CliftTypes.h"
#include "revng/Support/Identifier.h"

#include "CliftBytecode.h"

// keep this order
#include "revng/Clift/CliftAttributes.h"

#define GET_TYPEDEF_CLASSES
#include "revng/Clift/CliftTypes.cpp.inc"

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

static mlir::ParseResult parseSimpleStringAttributeImpl(mlir::AsmParser &Parser,
                                                        llvm::StringRef Name,
                                                        std::string &Value) {
  if (Parser.parseOptionalKeyword(Name).succeeded()) {
    if (Parser.parseString(&Value).failed())
      return mlir::failure();
  }
  return mlir::success();
}

static void printSimpleStringAttributeImpl(mlir::AsmPrinter &Printer,
                                           llvm::StringRef Name,
                                           llvm::StringRef Value) {
  if (not Value.empty()) {
    Printer << " " << Name << " ";
    printString(Printer, Value);
  }
}

static mlir::ParseResult parseName(mlir::AsmParser &Parser,
                                   std::string &Value) {
  return parseSimpleStringAttributeImpl(Parser, "as", Value);
}

static void printName(mlir::AsmPrinter &Printer, llvm::StringRef Value) {
  return printSimpleStringAttributeImpl(Printer, "as", Value);
}

static mlir::ParseResult parseComment(mlir::AsmParser &Parser,
                                      std::string &Comment) {
  return parseSimpleStringAttributeImpl(Parser, "comment", Comment);
}

static void printComment(mlir::AsmPrinter &Printer, llvm::StringRef Value) {
  return printSimpleStringAttributeImpl(Printer, "comment", Value);
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

//===------------------------------ VoidType ------------------------------===//

mlir::LogicalResult VoidType::verify(EmitErrorType EmitError, bool IsConst) {
  return mlir::success();
}

bool VoidType::getAlias(llvm::raw_ostream &OS) const {
  if (getIsConst())
    return false;

  OS << "void";
  return true;
}

ValueType VoidType::addConst() const {
  if (isConst())
    return *this;

  return get(getContext(), /*IsConst=*/true);
}

ValueType VoidType::removeConst() const {
  if (not isConst())
    return *this;

  return get(getContext(), /*IsConst=*/false);
}

template<std::same_as<VoidType>>
static VoidType readType(mlir::DialectBytecodeReader &Reader) {
  return VoidType::get(Reader.getContext());
}

static void writeType(VoidType Type, mlir::DialectBytecodeWriter &Writer) {
}

//===----------------------------- IntegerType ----------------------------===//

mlir::LogicalResult IntegerType::verify(EmitErrorType EmitError,
                                        IntegerKind Kind,
                                        uint64_t Size,
                                        bool IsConst) {
  if (Size == 0)
    return EmitError() << "Integer types must have non-zero size.";

  return mlir::success();
}

static llvm::StringRef getIntegerKindAlias(IntegerKind Kind) {
  switch (Kind) {
  case IntegerKind::Signed:
    return "int";
  case IntegerKind::Unsigned:
    return "uint";
  default:
    return stringifyIntegerKind(Kind);
  }
}

bool IntegerType::getAlias(llvm::raw_ostream &OS) const {
  if (getIsConst())
    return false;

  OS << getIntegerKindAlias(getKind()) << getSize() * 8 << "_t";
  return true;
}

uint64_t IntegerType::getObjectSize() const {
  return getSize();
}

ValueType IntegerType::addConst() const {
  if (isConst())
    return *this;

  return get(getContext(), getKind(), getSize(), /*IsConst=*/true);
}

ValueType IntegerType::removeConst() const {
  if (not isConst())
    return *this;

  return get(getContext(), getKind(), getSize(), /*IsConst=*/false);
}

template<std::same_as<IntegerType>>
static IntegerType readType(mlir::DialectBytecodeReader &Reader) {
  uint64_t KindInteger;
  if (Reader.readVarInt(KindInteger).failed())
    return {};

  auto Kind = symbolizeIntegerKind(KindInteger);
  if (not Kind)
    return {};

  uint64_t Size;
  if (Reader.readVarInt(Size).failed())
    return {};

  bool Const;
  if (readBool(Const, Reader).failed())
    return {};

  return IntegerType::get(Reader.getContext(), *Kind, Size, Const);
}

static void writeType(IntegerType Type, mlir::DialectBytecodeWriter &Writer) {
  Writer.writeVarInt(static_cast<uint64_t>(Type.getKind()));
  Writer.writeVarInt(Type.getSize());
  writeBool(Type.getIsConst(), Writer);
}

//===------------------------------ FloatType -----------------------------===//

mlir::LogicalResult
FloatType::verify(EmitErrorType EmitError, uint64_t Size, bool IsConst) {
  if (Size == 0)
    return EmitError() << "Floating point types must have non-zero size.";

  return mlir::success();
}

bool FloatType::getAlias(llvm::raw_ostream &OS) const {
  if (getIsConst())
    return false;

  OS << "float" << getSize() * 8 << "_t";
  return true;
}

uint64_t FloatType::getObjectSize() const {
  return getSize();
}

ValueType FloatType::addConst() const {
  if (isConst())
    return *this;

  return get(getContext(), getSize(), /*IsConst=*/true);
}

ValueType FloatType::removeConst() const {
  if (not isConst())
    return *this;

  return get(getContext(), getSize(), /*IsConst=*/false);
}

template<std::same_as<FloatType>>
static FloatType readType(mlir::DialectBytecodeReader &Reader) {
  uint64_t Size;
  if (Reader.readVarInt(Size).failed())
    return {};

  bool Const;
  if (readBool(Const, Reader).failed())
    return {};

  return FloatType::get(Reader.getContext(), Size, Const);
}

static void writeType(FloatType Type, mlir::DialectBytecodeWriter &Writer) {
  Writer.writeVarInt(Type.getSize());
  writeBool(Type.getIsConst(), Writer);
}

//===----------------------------- PointerType ----------------------------===//

mlir::LogicalResult PointerType::verify(EmitErrorType EmitError,
                                        mlir::Type ElementType,
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

uint64_t PointerType::getObjectSize() const {
  return getPointerSize();
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
  mlir::Type PointeeType;
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
                                      mlir::Type ElementType,
                                      uint64_t ElementCount) {
  if (not isObjectType(ElementType))
    return EmitError() << "Array type element type must be an object type.";
  if (ElementCount == 0)
    return EmitError() << "Array type must have no less than one element.";

  return mlir::success();
}

uint64_t ArrayType::getObjectSize() const {
  return getElementsCount() * clift::getObjectSize(getElementType());
}

bool ArrayType::isConst() const {
  return clift::isConst(getElementType());
}

ValueType ArrayType::addConst() const {
  auto ElementT = getElementType();
  auto NewElementT = clift::addConst(ElementT);

  if (ElementT == NewElementT)
    return *this;

  return get(getContext(), NewElementT, getElementsCount());
}

ValueType ArrayType::removeConst() const {
  auto ElementT = getElementType();
  auto NewElementT = clift::removeConst(ElementT);

  if (ElementT == NewElementT)
    return *this;

  return get(getContext(), NewElementT, getElementsCount());
}

template<std::same_as<clift::ArrayType>>
static clift::ArrayType readType(mlir::DialectBytecodeReader &Reader) {
  mlir::Type ElementType;
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

uint64_t EnumType::getObjectSize() const {
  return clift::getObjectSize(getUnderlyingType());
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
  if (parseName(Parser, Name).failed())
    return {};

  if (Parser.parseColon().failed())
    return {};

  mlir::Type UnderlyingType;
  if (Parser.parseType(UnderlyingType).failed())
    return {};

  llvm::SmallVector<clift::EnumFieldAttr> Enumerators;
  auto ParseEnumerator = [&]() -> mlir::ParseResult {
    std::string Handle;
    if (Parser.parseString(&Handle).failed())
      return mlir::failure();

    std::string Name;
    if (parseName(Parser, Name).failed())
      return mlir::failure();

    if (Parser.parseColon().failed())
      return mlir::failure();

    uint64_t Value;
    if (Parser.parseInteger(Value).failed())
      return mlir::failure();

    std::string Comment;
    if (parseComment(Parser, Comment).failed())
      return mlir::failure();

    auto NameAttr = makeNameAttr<clift::EnumFieldAttr>(Parser.getContext(),
                                                       Handle,
                                                       Name);

    auto CommentA = makeCommentAttr<clift::EnumFieldAttr>(Parser.getContext(),
                                                          Handle,
                                                          Comment);

    auto Attr = clift::EnumFieldAttr::getChecked(getEmitError(Parser, Loc),
                                                 Parser.getContext(),
                                                 llvm::StringRef(Handle),
                                                 NameAttr,
                                                 CommentA,
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

  std::string Comment;
  if (parseComment(Parser, Comment).failed())
    return {};

  if (Parser.parseGreater().failed())
    return {};

  auto NameAttr = makeNameAttr<EnumAttr>(Parser.getContext(), Handle, Name);
  auto CommentAttr = makeCommentAttr<EnumAttr>(Parser.getContext(),
                                               Handle,
                                               Comment);
  auto Attr = EnumAttr::getChecked(getEmitError(Parser, Loc),
                                   Parser.getContext(),
                                   llvm::StringRef(Handle),
                                   NameAttr,
                                   CommentAttr,
                                   UnderlyingType,
                                   llvm::ArrayRef(Enumerators));

  if (not Attr)
    return {};

  return EnumType::get(Attr);
}

void EnumType::print(mlir::AsmPrinter &Printer) const {
  Printer << "<";
  printString(Printer, getHandle());
  printName(Printer, getName());

  Printer << " : ";
  Printer.printType(getUnderlyingType());

  Printer << " {";
  if (not getFields().empty()) {
    for (auto [I, E] : llvm::enumerate(getFields())) {
      if (I != 0)
        Printer << ',';

      Printer << "\n  ";
      printString(Printer, E.getHandle());
      printName(Printer, E.getName());
      Printer << " : " << E.getRawValue();
      printComment(Printer, E.getComment());
    }
    Printer << '\n';
  }
  Printer << "}";
  printComment(Printer, getComment());
  Printer << ">";
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
  if (parseName(Parser, Name).failed())
    return {};

  if (Parser.parseColon().failed())
    return {};

  mlir::Type UnderlyingType;
  if (Parser.parseType(UnderlyingType).failed())
    return {};

  std::string Comment;
  if (parseComment(Parser, Comment).failed())
    return {};

  if (Parser.parseGreater().failed())
    return {};

  auto NameAttr = makeNameAttr<TypedefAttr>(Parser.getContext(), Handle, Name);
  auto CommentAttr = makeCommentAttr<TypedefAttr>(Parser.getContext(),
                                                  Handle,
                                                  Comment);
  auto Attr = TypedefAttr::getChecked(getEmitError(Parser, Loc),
                                      Parser.getContext(),
                                      llvm::StringRef(Handle),
                                      NameAttr,
                                      CommentAttr,
                                      UnderlyingType);

  return TypedefType::get(Attr);
}

void TypedefType::print(mlir::AsmPrinter &Printer) const {
  Printer << "<";
  printString(Printer, getHandle());
  printName(Printer, getName());

  Printer << " : ";
  Printer.printType(getUnderlyingType());

  printComment(Printer, getComment());
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

mlir::LogicalResult
FunctionType::verify(EmitErrorType EmitError,
                     llvm::StringRef Handle,
                     MutableStringAttr Name,
                     MutableStringAttr Comment,
                     mlir::Type ReturnType,
                     llvm::ArrayRef<mlir::Type> Args,
                     llvm::ArrayRef<CAttributeAttr> Attributes) {
  for (mlir::Type T : Args) {
    if (not isObjectType(T))
      return EmitError() << "Function parameter type must be an object type";
    if (isArrayType(T))
      return EmitError() << "Function parameter type may not be an array type";
  }

  auto EmitReturnError = [&]() -> mlir::InFlightDiagnostic {
    return EmitError() << "Function return type ";
  };

  if (verifyReturnType(EmitReturnError, ReturnType).failed())
    return mlir::failure();

  return mlir::success();
}

bool FunctionType::getAlias(llvm::raw_ostream &OS) const {
  return getDefinedTypeAlias(*this, OS);
}

bool FunctionType::isConst() const {
  return false;
}

ValueType FunctionType::addConst() const {
  return *this;
}

ValueType FunctionType::removeConst() const {
  return *this;
}

llvm::ArrayRef<mlir::Type> FunctionType::getResultTypes() const {
  return ArrayRef<Type>(getImpl()->return_type);
}

static mlir::LogicalResult
parseAttributeArrayImpl(mlir::AsmParser &Parser,
                        llvm::SmallVector<mlir::clift::CAttributeAttr> &Out) {
  revng_assert(Out.empty());

  mlir::ArrayAttr RawAttributes;
  auto MaybeResult = Parser.parseOptionalAttribute(RawAttributes);
  if (not MaybeResult.has_value())
    return mlir::success(); // no attributes - no problem

  else if (MaybeResult.value().failed())
    return mlir::failure();

  for (mlir::Attribute RawAttribute : RawAttributes) {
    if (auto CAttribute = mlir::cast<mlir::clift::CAttributeAttr>(RawAttribute))
      Out.emplace_back(CAttribute);
    else
      return mlir::failure();
  }

  return mlir::success();
}

mlir::Type FunctionType::parse(mlir::AsmParser &Parser) {
  mlir::SMLoc Loc = Parser.getCurrentLocation();

  if (Parser.parseLess().failed())
    return {};

  std::string Handle;
  if (Parser.parseString(&Handle).failed())
    return {};

  std::string Name;
  if (parseName(Parser, Name).failed())
    return {};

  if (Parser.parseColon().failed())
    return {};

  mlir::Type ReturnType;
  if (Parser.parseType(ReturnType).failed())
    return {};

  llvm::SmallVector<mlir::Type> ParameterTypes;
  const auto ParseParameterType = [&Parser,
                                   &ParameterTypes]() -> mlir::ParseResult {
    mlir::Type ParameterType;
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

  llvm::SmallVector<mlir::clift::CAttributeAttr> Attributes;
  if (parseAttributeArrayImpl(Parser, Attributes).failed())
    return {};

  std::string Comment;
  if (parseComment(Parser, Comment).failed())
    return {};

  if (Parser.parseGreater().failed())
    return {};

  auto NameAttr = makeNameAttr<FunctionType>(Parser.getContext(), Handle, Name);
  auto CommentAttr = makeCommentAttr<FunctionType>(Parser.getContext(),
                                                   Handle,
                                                   Comment);
  return FunctionType::getChecked(getEmitError(Parser, Loc),
                                  Parser.getContext(),
                                  llvm::StringRef(Handle),
                                  NameAttr,
                                  CommentAttr,
                                  ReturnType,
                                  llvm::ArrayRef(ParameterTypes),
                                  llvm::ArrayRef(Attributes));
}

void FunctionType::print(mlir::AsmPrinter &Printer) const {
  Printer << "<";
  printString(Printer, getHandle());
  printName(Printer, getName());

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

  if (not getCAttributes().empty()) {
    llvm::ArrayRef<mlir::Attribute> Attributes = { getCAttributes().begin(),
                                                   getCAttributes().end() };

    Printer << '\n';
    Printer.printAttribute(mlir::ArrayAttr::get(getContext(), Attributes));
  }

  printComment(Printer, getComment());

  Printer << ">";
}

template<std::same_as<clift::FunctionType>>
static clift::FunctionType readType(mlir::DialectBytecodeReader &Reader) {
  auto ReadType = [&](mlir::Type &Type) -> mlir::LogicalResult {
    return Reader.readType(Type);
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

  llvm::SmallVector<mlir::clift::CAttributeAttr> Attributes;
  auto ReadAttribute = [&](mlir::clift::CAttributeAttr &Attribute) {
    return Reader.readAttribute(Attribute);
  };
  if (Reader.readList(Attributes, ReadAttribute).failed())
    return {};

  llvm::StringRef Comment;
  if (Reader.readString(Comment).failed())
    return {};

  mlir::MLIRContext *Context = Reader.getContext();
  auto NameAttr = makeNameAttr<FunctionType>(Context, Handle, Name);
  auto CommentAttr = makeCommentAttr<FunctionType>(Context, Handle, Comment);
  return clift::FunctionType::getChecked(getEmitError(Reader),
                                         Context,
                                         Handle,
                                         NameAttr,
                                         CommentAttr,
                                         ReturnType,
                                         std::move(ParameterTypes),
                                         llvm::ArrayRef(Attributes));
}

static void writeType(clift::FunctionType Type,
                      mlir::DialectBytecodeWriter &Writer) {
  Writer.writeOwnedString(Type.getHandle());
  Writer.writeOwnedString(Type.getName());
  Writer.writeType(Type.getReturnType());
  Writer.writeList(Type.getArgumentTypes(),
                   [&](mlir::Type Type) { Writer.writeType(Type); });

  Writer.writeList(Type.getCAttributes(),
                   [&](mlir::clift::CAttributeAttr Attribute) {
                     Writer.writeAttribute(Attribute);
                   });

  Writer.writeOwnedString(Type.getComment());
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
  if (parseName(Parser, Name).failed())
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
    if (parseName(Parser, Name))
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

    mlir::Type Type;
    if (Parser.parseType(Type).failed())
      return mlir::failure();

    std::string Comment;
    if (parseComment(Parser, Comment).failed())
      return {};

    auto NameAttr = makeNameAttr<FieldAttr>(Parser.getContext(), Handle, Name);
    auto CommentAttr = makeCommentAttr<FieldAttr>(Parser.getContext(),
                                                  Handle,
                                                  Comment);
    auto Attr = FieldAttr::getChecked(getEmitError(Parser, FieldLoc),
                                      Parser.getContext(),
                                      llvm::StringRef(Handle),
                                      NameAttr,
                                      CommentAttr,
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

  llvm::SmallVector<mlir::clift::CAttributeAttr> Attributes;
  if (parseAttributeArrayImpl(Parser, Attributes).failed())
    return {};

  std::string Comment;
  if (parseComment(Parser, Comment).failed())
    return {};

  if (Parser.parseGreater().failed())
    return {};

  auto GetCompleteType = [&](const auto &...Args) -> TypeT {
    auto NameAttr = makeNameAttr<AttrT>(Parser.getContext(), Handle, Name);
    auto CommentAttr = makeCommentAttr<AttrT>(Parser.getContext(),
                                              Handle,
                                              Comment);
    auto Attr = AttrT::getChecked(getEmitError(Parser, Loc),
                                  Parser.getContext(),
                                  llvm::StringRef(Handle),
                                  NameAttr,
                                  CommentAttr,
                                  Args...,
                                  llvm::ArrayRef(Fields),
                                  llvm::ArrayRef(Attributes));

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
  printName(Printer, Type.getName());

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
      printName(Printer, S.getName());

      Printer << " :";

      if constexpr (IsStruct) {
        Printer << " offset(" << S.getOffset() << ")";
      }

      Printer << ' ' << S.getType();
      printComment(Printer, S.getComment());
    }
    Printer << '\n';
  }
  Printer << "}";

  if (not Type.getDefinition().getCAttributes().empty()) {
    llvm::SmallVector<mlir::Attribute> Attributes;
    for (mlir::Attribute Attribute : Type.getDefinition().getCAttributes())
      Attributes.emplace_back(Attribute);

    Printer << '\n';
    Printer.printAttribute(mlir::ArrayAttr::get(Type.getContext(), Attributes));
  }

  printComment(Printer, Type.getComment());

  Printer << ">";
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

    mlir::Type Type;
    if (Reader.readType(Type).failed())
      return mlir::failure();

    llvm::StringRef Comment;
    if (Reader.readString(Comment).failed())
      return mlir::failure();

    auto NameAttr = makeNameAttr<clift::FieldAttr>(Reader.getContext(),
                                                   Handle,
                                                   Name);
    auto CommentAttr = makeCommentAttr<clift::FieldAttr>(Reader.getContext(),
                                                         Handle,
                                                         Comment);

    Field = clift::FieldAttr::getChecked(getEmitError(Reader),
                                         Reader.getContext(),
                                         llvm::StringRef(Handle),
                                         NameAttr,
                                         CommentAttr,
                                         Offset,
                                         Type);

    return mlir::success(static_cast<bool>(Field));
  };

  llvm::SmallVector<clift::FieldAttr> Fields;
  if (Reader.readList(Fields, ReadField).failed())
    return {};

  llvm::SmallVector<mlir::clift::CAttributeAttr> Attributes;
  auto ReadAttribute = [&](mlir::clift::CAttributeAttr &Attribute) {
    return Reader.readAttribute(Attribute);
  };
  if (Reader.readList(Attributes, ReadAttribute).failed())
    return {};

  llvm::StringRef Comment;
  if (Reader.readString(Comment).failed())
    return {};

  auto GetCompleteType = [&](const auto &...Args) -> TypeT {
    auto NameAttr = makeNameAttr<AttrT>(Reader.getContext(), Handle, Name);
    auto CommentAttr = makeCommentAttr<AttrT>(Reader.getContext(),
                                              Handle,
                                              Comment);
    auto Attr = AttrT::getChecked(getEmitError(Reader),
                                  Reader.getContext(),
                                  llvm::StringRef(Handle),
                                  NameAttr,
                                  CommentAttr,
                                  Args...,
                                  llvm::ArrayRef(Fields),
                                  Attributes);

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
    Writer.writeOwnedString(Field.getComment());
  });

  Writer.writeList(Type.getCAttributes(),
                   [&](mlir::clift::CAttributeAttr Attribute) {
                     Writer.writeAttribute(Attribute);
                   });

  Writer.writeOwnedString(Type.getComment());
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

//===------------------------------ Typedefs ------------------------------===//

TypedefDecomposition clift::decomposeTypedef(mlir::Type Type) {
  bool HasConstTypedef = false;

  while (true) {
    auto T = mlir::dyn_cast<TypedefType>(Type);
    if (not T)
      break;

    Type = T.getUnderlyingType();
    HasConstTypedef |= T.getIsConst();
  }

  return { Type, HasConstTypedef };
}

mlir::Type clift::unwrapTypedefs(mlir::Type Type) {
  return decomposeTypedef(Type).Type;
}

mlir::Type clift::collapseTypedefs(mlir::Type Type) {
  auto [UnderlyingType, HasConstTypedef] = decomposeTypedef(Type);

  if (HasConstTypedef)
    UnderlyingType = addConst(UnderlyingType);

  return UnderlyingType;
}

//===---------------------------- CV-Qualifiers ---------------------------===//

bool clift::isConst(mlir::Type Type) {
  if (auto ValueT = mlir::dyn_cast<clift::ValueType>(Type))
    return ValueT.isConst();

  return false;
}

mlir::Type clift::addConst(mlir::Type Type) {
  if (auto ValueT = mlir::dyn_cast<clift::ValueType>(Type))
    Type = ValueT.addConst();

  return Type;
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

  return removeConst(LhsVT) == removeConst(RhsVT);
}

//===-------------------------- Object type size --------------------------===//

uint64_t clift::getObjectSize(mlir::Type Type) {
  return mlir::cast<ObjectType>(unwrapTypedefs(Type)).getObjectSize();
}

uint64_t clift::getObjectSizeOrZero(mlir::Type Type) {
  if (auto ValueT = mlir::dyn_cast<ObjectType>(unwrapTypedefs(Type)))
    return ValueT.getObjectSize();

  return 0;
}

//===--------------------------- Type categories --------------------------===//

bool clift::isModifiableType(mlir::Type Type) {
  auto [UnderlyingType, HasConst] = decomposeTypedef(Type);
  return not HasConst and not isConst(UnderlyingType);
}

IntegerType clift::getUnderlyingIntegerType(mlir::Type Type) {
  Type = unwrapTypedefs(Type);

  if (auto T = mlir::dyn_cast<IntegerType>(Type))
    return T;

  if (auto T = mlir::dyn_cast<EnumType>(Type))
    return mlir::cast<IntegerType>(collapseTypedefs(T.getUnderlyingType()));

  return nullptr;
}

bool clift::isCompleteType(mlir::Type Type) {
  Type = unwrapTypedefs(Type);

  while (auto Array = mlir::dyn_cast<ArrayType>(Type))
    Type = unwrapTypedefs(Array.getElementType());

  if (auto Class = mlir::dyn_cast<ClassType>(Type))
    return Class.isComplete();

  return true;
}

bool clift::isVoid(mlir::Type Type) {
  return mlir::isa<VoidType>(unwrapTypedefs(Type));
}

bool clift::isScalarType(mlir::Type Type) {
  Type = unwrapTypedefs(Type);
  return mlir::isa<IntegerType, FloatType, EnumType, PointerType>(Type);
}

IntegerType clift::getPrimitiveIntegerType(mlir::Type Type) {
  return mlir::dyn_cast<IntegerType>(unwrapTypedefs(Type));
}

bool clift::isPrimitiveIntegerType(mlir::Type Type) {
  return mlir::isa<IntegerType>(unwrapTypedefs(Type));
}

bool clift::isIntegerType(mlir::Type Type) {
  return mlir::isa<IntegerType, EnumType>(unwrapTypedefs(Type));
}

bool clift::isBooleanType(mlir::Type Type) {
  auto T = getPrimitiveIntegerType(Type);
  return T and T.isSigned();
}

bool clift::isFloatType(mlir::Type Type) {
  return mlir::isa<FloatType>(unwrapTypedefs(Type));
}

PointerType clift::getPointerType(mlir::Type Type) {
  return mlir::dyn_cast<PointerType>(unwrapTypedefs(Type));
}

bool clift::isPointerType(mlir::Type Type) {
  return mlir::isa<PointerType>(unwrapTypedefs(Type));
}

bool clift::isObjectType(mlir::Type Type) {
  return mlir::isa<ArrayType,
                   ClassType,
                   EnumType,
                   FloatType,
                   IntegerType,
                   PointerType>(unwrapTypedefs(Type));
}

bool clift::isArrayType(mlir::Type Type) {
  return mlir::isa<ArrayType>(unwrapTypedefs(Type));
}

bool clift::isEnumType(mlir::Type Type) {
  return mlir::isa<EnumType>(unwrapTypedefs(Type));
}

bool clift::isClassType(mlir::Type Type) {
  return mlir::isa<ClassType>(unwrapTypedefs(Type));
}

bool clift::isFunctionType(mlir::Type Type) {
  return mlir::isa<clift::FunctionType>(unwrapTypedefs(Type));
}

clift::FunctionType
clift::getFunctionOrFunctionPointerFunctionType(mlir::Type Type) {
  Type = unwrapTypedefs(Type);

  if (auto P = mlir::dyn_cast<PointerType>(Type))
    Type = unwrapTypedefs(P.getPointeeType());

  return mlir::dyn_cast<clift::FunctionType>(Type);
}

mlir::LogicalResult clift::verifyReturnType(EmitErrorType EmitError,
                                            mlir::Type ReturnType) {
  ReturnType = unwrapTypedefs(ReturnType);

  if (not isVoid(ReturnType)) {
    if (not isObjectType(ReturnType))
      return EmitError() << "must be an object type or void.";

    if (isArrayType(ReturnType))
      return EmitError() << "cannot be an array type.";
  }

  return mlir::success();
}

//===---------------------------- CliftDialect ----------------------------===//

void CliftDialect::registerTypes() {
  addTypes</* Include the auto-generated clift types */
#define GET_TYPEDEF_LIST
#include "revng/Clift/CliftTypes.cpp.inc"
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
  Void,
  Int,
  Float,
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
  case CliftTypeKind::Void:
    return ::readType<clift::VoidType>(Reader);
  case CliftTypeKind::Int:
    return ::readType<clift::IntegerType>(Reader);
  case CliftTypeKind::Float:
    return ::readType<clift::FloatType>(Reader);
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
  if (auto T = mlir::dyn_cast<clift::VoidType>(Type))
    return Write(T, CliftTypeKind::Void);
  if (auto T = mlir::dyn_cast<clift::IntegerType>(Type))
    return Write(T, CliftTypeKind::Int);
  if (auto T = mlir::dyn_cast<clift::FloatType>(Type))
    return Write(T, CliftTypeKind::Float);
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
