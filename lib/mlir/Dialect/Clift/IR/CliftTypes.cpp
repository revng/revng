//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <string>

#include "llvm/ADT/SmallSet.h"

#include "revng-c/mlir/Dialect/Clift/IR/CliftTypes.h"
// keep this order
#include "revng/Model/PrimitiveType.h"

#include "revng-c/mlir/Dialect/Clift/IR/CliftAttributes.h"

#include "CliftParser.h"
#include "CliftStorage.h"

#define GET_TYPEDEF_CLASSES
#include "revng-c/mlir/Dialect/Clift/IR/CliftOpsTypes.cpp.inc"

using EmitErrorType = llvm::function_ref<mlir::InFlightDiagnostic()>;

//******************************** CliftDialect ********************************

void mlir::clift::CliftDialect::registerTypes() {
  addTypes<ScalarTupleType, /* Include the auto-generated clift types */
#define GET_TYPEDEF_LIST
#include "revng-c/mlir/Dialect/Clift/IR/CliftOpsTypes.cpp.inc"
           /* End of types list */>();
}

/// Parse a type registered to this dialect
mlir::Type
mlir::clift::CliftDialect::parseType(mlir::DialectAsmParser &Parser) const {
  const llvm::SMLoc TypeLoc = Parser.getCurrentLocation();

  llvm::StringRef Mnemonic;
  if (mlir::Type GenType;
      generatedTypeParser(Parser, &Mnemonic, GenType).has_value())
    return GenType;

  if (Mnemonic == ScalarTupleType::getMnemonic())
    return ScalarTupleType::parse(Parser);

  Parser.emitError(TypeLoc) << "unknown  type `" << Mnemonic << "` in dialect `"
                            << getNamespace() << "`";
  return {};
}

/// Print a type registered to this dialect
void mlir::clift::CliftDialect::printType(mlir::Type Type,
                                          mlir::DialectAsmPrinter &Printer)
  const {

  if (mlir::succeeded(generatedTypePrinter(Type, Printer)))
    return;

  if (auto T = mlir::dyn_cast<ScalarTupleType>(Type))
    return T.print(Printer);
}

static constexpr model::PrimitiveType::PrimitiveKindType
kindToKind(mlir::clift::PrimitiveKind kind) {
  return static_cast<model::PrimitiveType::PrimitiveKindType>(kind);
}

using Primitive = model::PrimitiveType::PrimitiveKindType;
using namespace mlir::clift;
static_assert(Primitive::Float == kindToKind(PrimitiveKind::FloatKind));
static_assert(Primitive::Void == kindToKind(PrimitiveKind::VoidKind));
static const auto
  PointerOrNumber = kindToKind(PrimitiveKind::PointerOrNumberKind);
static_assert(Primitive::PointerOrNumber == PointerOrNumber);
static_assert(Primitive::Unsigned == kindToKind(PrimitiveKind::UnsignedKind));
static_assert(Primitive::Generic == kindToKind(PrimitiveKind::GenericKind));
static_assert(Primitive::Signed == kindToKind(PrimitiveKind::SignedKind));
static_assert(Primitive::Number == kindToKind(PrimitiveKind::NumberKind));

mlir::LogicalResult
mlir::clift::PrimitiveType::verify(EmitErrorType EmitError,
                                   mlir::clift::PrimitiveKind kind,
                                   uint64_t size,
                                   BoolAttr IsConst) {

  model::PrimitiveType Type(kindToKind(kind), size);
  if (not Type.verify()) {
    return EmitError() << "primitive type verify failed";
  }
  return mlir::success();
}

std::string mlir::clift::EnumAttr::getAlias() const {
  return getName().str();
}

std::string mlir::clift::DefinedType::getAlias() const {
  if (auto Casted = getElementType().dyn_cast<mlir::clift::AliasableAttr>()) {
    if (Casted.getAlias().empty())
      return "";
    return isConst() ? "const_" + Casted.getAlias() : Casted.getAlias();
  }
  return "";
}

std::string mlir::clift::PrimitiveType::getAlias() const {
  model::PrimitiveType Type(kindToKind(getKind()), getByteSize());
  return isConst() ? std::string("const_") + serializeToString(Type.name()) :
                     serializeToString(Type.name());
}

std::string mlir::clift::TypedefAttr::getAlias() const {
  return getName().str();
}

std::string mlir::clift::FunctionAttr::getAlias() const {
  return getName().str();
}

llvm::StringRef mlir::clift::DefinedType::name() {
  return getElementType().name();
}

uint64_t mlir::clift::DefinedType::id() {
  return getElementType().id();
}

uint64_t mlir::clift::ArrayType::getByteSize() const {
  return getElementType().getByteSize() * getElementsCount();
}

uint64_t mlir::clift::DefinedType::getByteSize() const {
  return getElementType().cast<mlir::clift::SizedType>().getByteSize();
}

uint64_t mlir::clift::EnumAttr::getByteSize() const {
  return getUnderlyingType().cast<mlir::clift::PrimitiveType>().getSize();
}

uint64_t mlir::clift::PointerType::getByteSize() const {
  return getPointerSize();
}

uint64_t mlir::clift::TypedefAttr::getByteSize() const {
  return getUnderlyingType().getByteSize();
}

uint64_t mlir::clift::FunctionAttr::getByteSize() const {
  return 0;
}

mlir::LogicalResult PointerType::verify(EmitErrorType emitError,
                                        mlir::clift::ValueType element_type,
                                        uint64_t pointer_size,
                                        BoolAttr IsConst) {
  switch (pointer_size) {
  case 4:
  case 8:
    break;
  default:
    return emitError() << "invalid pointer size: " << pointer_size;
  }
  return mlir::success();
}

mlir::LogicalResult
DefinedType::verify(EmitErrorType emitError,
                    mlir::clift::TypeDefinition element_type,
                    BoolAttr IsConst) {
  return mlir::success();
}

mlir::LogicalResult ArrayType::verify(EmitErrorType emitError,
                                      mlir::clift::ValueType element_type,
                                      uint64_t count,
                                      BoolAttr IsConst) {
  if (count == 0) {
    return emitError() << "array type cannot have zero elements";
  }
  if (element_type.getByteSize() == 0) {
    return emitError() << "array type cannot have size zero. Did you created a "
                          "array type with a uninitialized struct or union "
                          "type inside?";
  }
  return mlir::success();
}

static mlir::clift::TypedefAttr getTypedefAttr(mlir::Type Type) {
  if (auto D = mlir::dyn_cast<mlir::clift::DefinedType>(Type))
    return mlir::dyn_cast<mlir::clift::TypedefAttr>(D.getElementType());
  return nullptr;
}

static mlir::Type dealias(mlir::Type Type) {
  while (auto Attr = getTypedefAttr(Type))
    Type = Attr.getUnderlyingType();
  return Type;
}

using namespace mlir::clift;
mlir::LogicalResult
EnumAttr::verify(EmitErrorType emitError,
                 uint64_t ID,
                 llvm::StringRef,
                 mlir::Type UnderlyingType,
                 llvm::ArrayRef<mlir::clift::EnumFieldAttr> Fields) {
  UnderlyingType = dealias(UnderlyingType);

  if (not UnderlyingType.isa<mlir::clift::PrimitiveType>())
    return emitError() << "type of enum must be a primitive type";

  const auto PrimitiveType = UnderlyingType.cast<mlir::clift::PrimitiveType>();
  const uint64_t BitWidth = PrimitiveType.getSize() * 8;

  if (Fields.empty())
    return emitError() << "enum requires at least one field";

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
    return emitError() << "enum underlying type must be an integral type";
  }

  uint64_t LastValue = 0;
  bool CheckEqual = false;

  for (const auto &Field : Fields) {
    const uint64_t Value = Field.getRawValue();

    const auto UsingSigned = [&](auto Callable, const auto... V) {
      return IsSigned ? Callable(static_cast<int64_t>(V)...) : Callable(V...);
    };

    const auto CheckSigned =
      [emitError](const auto Value,
                  const auto MinValue,
                  const auto MaxValue) -> mlir::LogicalResult {
      if (Value < MinValue)
        return emitError() << "enum field " << Value
                           << " is less than the min value of the "
                              "underlying type "
                           << MinValue;

      if (Value > MaxValue)
        return emitError() << "enum field " << Value
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
      return emitError() << "enum fields must be strictly ordered by their "
                            "unsigned values";

    LastValue = Value;
    CheckEqual = true;
  }

  return mlir::success();
}

void UnionTypeAttr::walkImmediateSubElements(function_ref<void(Attribute)>
                                               WalkAttr,
                                             function_ref<void(Type)> WalkType)
  const {
  if (not getImpl()->isInitialized())
    return;
  for (auto Field : getFields())
    WalkAttr(Field);
}

mlir::Attribute
UnionTypeAttr::replaceImmediateSubElements(llvm::ArrayRef<mlir::Attribute>,
                                           llvm::ArrayRef<mlir::Type>) const {
  revng_abort("it does not make any sense to replace the elements of a "
              "defined union");
  return {};
}

void StructTypeAttr::walkImmediateSubElements(function_ref<void(Attribute)>
                                                WalkAttr,
                                              function_ref<void(Type)> WalkType)
  const {
  if (not getImpl()->isInitialized())
    return;
  for (auto Field : getFields())
    WalkAttr(Field);
}

mlir::Attribute
StructTypeAttr::replaceImmediateSubElements(llvm::ArrayRef<mlir::Attribute>,
                                            llvm::ArrayRef<mlir::Type>) const {
  revng_abort("it does not make any sense to replace the elements of a "
              "defined struct");
}

//****************************** ScalarTupleType *******************************

static bool isScalarType(mlir::Type Type) {
  Type = dealias(Type);

  if (auto T = mlir::dyn_cast<PrimitiveType>(Type))
    return T.getKind() != PrimitiveKind::VoidKind;

  if (auto T = mlir::dyn_cast<DefinedType>(Type))
    return mlir::isa<EnumAttr>(T.getElementType());

  return mlir::isa<PointerType>(Type);
}

mlir::LogicalResult ScalarTupleType::verify(const EmitErrorType EmitError,
                                            const uint64_t ID) {
  return mlir::success();
}

mlir::LogicalResult
ScalarTupleType::verify(const EmitErrorType EmitError,
                        const uint64_t ID,
                        const llvm::StringRef Name,
                        const llvm::ArrayRef<ScalarTupleElementAttr> Elements) {
  if (Elements.size() < 2)
    return EmitError() << "Scalar tuple types must have at least two elements";

  llvm::SmallSet<llvm::StringRef, 16> NameSet;
  for (auto Element : Elements) {
    if (not isScalarType(Element.getType()))
      return EmitError() << "Scalar tuple element types must be scalar types";

    if (not Element.getName().empty()) {
      if (not NameSet.insert(Element.getName()).second)
        return EmitError() << "Scalar tuple element names must be empty or "
                              "unique";
    }
  }

  return mlir::success();
}

ScalarTupleType ScalarTupleType::get(MLIRContext *const Context,
                                     const uint64_t ID) {
  return Base::get(Context, ID);
}

ScalarTupleType ScalarTupleType::getChecked(const EmitErrorType EmitError,
                                            MLIRContext *const Context,
                                            const uint64_t ID) {
  return get(Context, ID);
}

ScalarTupleType
ScalarTupleType::get(MLIRContext *const Context,
                     const uint64_t ID,
                     const llvm::StringRef Name,
                     const llvm::ArrayRef<ScalarTupleElementAttr> Elements) {
  auto Result = Base::get(Context, ID);
  Result.define(Name, Elements);
  return Result;
}

ScalarTupleType
ScalarTupleType::getChecked(const EmitErrorType EmitError,
                            MLIRContext *const Context,
                            const uint64_t ID,
                            const llvm::StringRef Name,
                            const llvm::ArrayRef<ScalarTupleElementAttr>
                              Elements) {
  if (failed(verify(EmitError, ID, Name, Elements)))
    return {};
  return get(Context, ID, Name, Elements);
}

void ScalarTupleType::define(const llvm::StringRef Name,
                             const llvm::ArrayRef<ScalarTupleElementAttr>
                               Elements) {
  LogicalResult Result = Base::mutate(Name, Elements);

  revng_assert(succeeded(Result)
               && "attempting to change the body of an already-initialized "
                  "type");
}

uint64_t ScalarTupleType::getId() const {
  return getImpl()->getID();
}

llvm::StringRef ScalarTupleType::getName() const {
  return getImpl()->getName();
}

llvm::ArrayRef<ScalarTupleElementAttr> ScalarTupleType::getElements() const {
  return getImpl()->getSubobjects();
}

bool ScalarTupleType::isComplete() const {
  return getImpl()->isInitialized();
}

uint64_t ScalarTupleType::getByteSize() const {
  uint64_t Size = 0;
  for (ScalarTupleElementAttr Element : getElements())
    Size += mlir::cast<ValueType>(Element.getType()).getByteSize();
  return Size;
}

std::string ScalarTupleType::getAlias() const {
  return getName().str();
}

mlir::BoolAttr ScalarTupleType::getIsConst() const {
  return BoolAttr::get(getContext(), false);
}

mlir::Type ScalarTupleType::parse(AsmParser &Parser) {
  return parseCompositeType<ScalarTupleType>(Parser, /*MinSubobjects=*/2);
}

void ScalarTupleType::print(AsmPrinter &Printer) const {
  return printCompositeType(Printer, *this);
}

void ScalarTupleType::walkImmediateSubElements(function_ref<void(Attribute)>
                                                 WalkAttr,
                                               function_ref<void(Type)>
                                                 WalkType) const {
  if (getImpl()->isInitialized()) {
    for (auto Element : getElements())
      WalkAttr(Element);
  }
}

mlir::Type ScalarTupleType::replaceImmediateSubElements(ArrayRef<Attribute>,
                                                        ArrayRef<Type>) const {
  revng_abort("it does not make any sense to replace the elements of a "
              "scalar tuple");
}
