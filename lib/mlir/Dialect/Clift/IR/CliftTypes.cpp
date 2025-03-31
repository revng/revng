//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <optional>
#include <string>

#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/SmallSet.h"

#include "revng/mlir/Dialect/Clift/IR/CliftTypes.h"

// keep this order
#include "revng/Model/Binary.h"
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

static bool getDefinedTypeAliasImpl(DefinedType Type,
                                    llvm::StringRef Kind,
                                    llvm::raw_ostream &OS) {
  auto IsIdentifierChar = [](char const X) {
    if (X == '_')
      return true;
    if ('a' <= X and X <= 'z')
      return true;
    if ('A' <= X and X <= 'Z')
      return true;
    if ('0' <= X and X <= '9')
      return true;
    return false;
  };

  if (not Type.getName().empty()) {
    OS << Type.getName();
  } else if (not Type.getHandle().empty()) {
    for (char C : Type.getHandle())
      OS << (IsIdentifierChar(C) ? C : '_');
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

//===---------------------------- PrimitiveType ---------------------------===//

static constexpr model::PrimitiveKind::Values
kindToKind(const PrimitiveKind Kind) {
  return static_cast<model::PrimitiveKind::Values>(Kind);
}

/// Test that kindToKind converts each clift::PrimitiveKind to the matching
/// model::PrimitiveKind. Use a switch converting in the opposite direction
/// in order to produce a warning if a new primitive kind is ever added.
static consteval bool testKindToKind() {
  PrimitiveKind UninitializedKind;
  const auto TestSwitch = [&](const model::PrimitiveKind::Values Kind) {
    switch (Kind) {
    case model::PrimitiveKind::Float:
      return PrimitiveKind::FloatKind;
    case model::PrimitiveKind::Generic:
      return PrimitiveKind::GenericKind;
    case model::PrimitiveKind::Number:
      return PrimitiveKind::NumberKind;
    case model::PrimitiveKind::PointerOrNumber:
      return PrimitiveKind::PointerOrNumberKind;
    case model::PrimitiveKind::Signed:
      return PrimitiveKind::SignedKind;
    case model::PrimitiveKind::Unsigned:
      return PrimitiveKind::UnsignedKind;
    case model::PrimitiveKind::Void:
      return PrimitiveKind::VoidKind;

    case model::PrimitiveKind::Invalid:
    case model::PrimitiveKind::Count:
      // Unreachable. This causes an error during constant evaluation.
      return UninitializedKind;
    }
  };

  for (int I = 0; I < static_cast<int>(model::PrimitiveKind::Count); ++I) {
    auto const Kind = static_cast<model::PrimitiveKind::Values>(I);
    if (Kind != model::PrimitiveKind::Invalid) {
      if (kindToKind(TestSwitch(Kind)) != Kind)
        return false;
    }
  }
  return true;
}
static_assert(testKindToKind());

mlir::LogicalResult PrimitiveType::verify(EmitErrorType EmitError,
                                          PrimitiveKind Kind,
                                          uint64_t Size,
                                          bool IsConst) {
  if (not model::PrimitiveType::make(kindToKind(Kind), Size)->verify())
    return EmitError() << "primitive type verify failed";

  return mlir::success();
}

bool PrimitiveType::getAlias(llvm::raw_ostream &OS) const {
  if (getIsConst())
    return false;

  OS << toString(model::PrimitiveType::getCName(kindToKind(getKind()),
                                                getByteSize()));
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
    uint64_t Value;
    if (Parser.parseInteger(Value).failed())
      return mlir::failure();

    std::string Name;
    if (mlir::parseCliftDebugName(Parser, Name).failed())
      return {};

    Enumerators.push_back(clift::EnumFieldAttr::get(Parser.getContext(),
                                                    Value,
                                                    std::move(Name)));

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

  return EnumType::get(Parser.getContext(),
                       Handle,
                       Name,
                       UnderlyingType,
                       Enumerators);
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

      Printer << "\n  " << E.getRawValue();
      mlir::printCliftDebugName(Printer, E.getName());
    }
    Printer << '\n';
  }
  Printer << "}>";
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

  return TypedefType::get(Parser.getContext(), Handle, Name, UnderlyingType);
}

void TypedefType::print(mlir::AsmPrinter &Printer) const {
  Printer << "<";
  printString(Printer, getHandle());
  mlir::printCliftDebugName(Printer, getName());

  Printer << " : ";
  Printer.printType(getUnderlyingType());
  Printer << ">";
}

//===---------------------------- FunctionType ----------------------------===//

mlir::LogicalResult FunctionType::verify(EmitErrorType EmitError,
                                         llvm::StringRef Handle,
                                         llvm::StringRef Name,
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

  if (not isReturnableType(R))
    return EmitError() << "Function return type must be void or a non-array "
                          "object type.";

  return mlir::success();
}

mlir::LogicalResult
FunctionType::verify(EmitErrorType EmitError,
                     llvm::StringRef Handle,
                     llvm::StringRef Name,
                     clift::ValueType ReturnType,
                     llvm::ArrayRef<clift::ValueType> Args) {
  if (not isReturnableType(ReturnType))
    return EmitError() << "Function return type must be void or a non-array "
                          "object type.";

  return mlir::success();
}

bool FunctionType::getAlias(llvm::raw_ostream &OS) const {
  return getDefinedTypeAlias(*this, OS);
}

llvm::ArrayRef<mlir::Type> FunctionType::getResultTypes() const {
  return ArrayRef<Type>(getImpl()->return_type);
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
    bool ParseColon = false;

    uint64_t Offset = 0;
    std::string Name;

    if constexpr (IsStruct) {
      if (Parser.parseKeyword("offset").failed())
        return {};

      if (Parser.parseLParen().failed())
        return {};

      if (Parser.parseInteger(Offset).failed())
        return mlir::failure();

      if (Parser.parseRParen().failed())
        return {};

      if (Parser.parseOptionalKeyword("as").succeeded()) {
        if (Parser.parseString(&Name).failed())
          return {};
      }

      ParseColon = true;
    } else {
      if (Parser.parseOptionalString(&Name).succeeded())
        ParseColon = true;
    }

    if (ParseColon and Parser.parseColon().failed())
      return mlir::failure();

    clift::ValueType Type;
    if (Parser.parseType(Type).failed())
      return mlir::failure();

    auto EmitError = [&] { return Parser.emitError(FieldLoc); };
    if (FieldAttr::verify(EmitError, Offset, Type, Name).failed())
      return mlir::failure();

    Fields.push_back(FieldAttr::get(Parser.getContext(),
                                    Offset,
                                    Type,
                                    std::move(Name)));

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
    auto EmitError = [&] { return Parser.emitError(Loc); };

    AttrT Attr = AttrT::getChecked(EmitError,
                                   Parser.getContext(),
                                   llvm::StringRef(Handle),
                                   llvm::StringRef(Name),
                                   Args...,
                                   llvm::ArrayRef(Fields));

    return TypeT::get(Parser.getContext(), Attr, /*IsConst=*/false);
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

      bool PrintColon = false;
      if constexpr (IsStruct) {
        Printer << "offset(" << S.getOffset() << ") ";
        PrintColon = true;
      }

      if (llvm::StringRef Name = S.getName(); not Name.empty()) {
        if constexpr (IsStruct) {
          Printer << "as ";
        }

        printString(Printer, Name);
        Printer << ' ';

        PrintColon = true;
      }

      if (PrintColon)
        Printer << ": ";

      Printer << S.getType();
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

bool clift::isPointerType(ValueType Type) {
  return mlir::isa<PointerType>(dealias(Type, /*IgnoreQualifiers=*/true));
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

bool clift::isReturnableType(ValueType ReturnType) {
  ReturnType = dealias(ReturnType);

  if (isObjectType(ReturnType))
    return not isArrayType(ReturnType);

  return isVoid(ReturnType);
}

clift::FunctionType
clift::getFunctionOrFunctionPointerFunctionType(ValueType Type) {
  Type = dealias(Type, /*IgnoreQualifiers=*/true);

  if (auto P = mlir::dyn_cast<PointerType>(Type))
    Type = dealias(P.getPointeeType(), /*IgnoreQualifiers=*/true);

  return mlir::dyn_cast<clift::FunctionType>(Type);
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
