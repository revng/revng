//
// Copyright (c) rev.ng Labs Srl. See LICENSE.md for details.
//

#include <string>

#include "revng-c/mlir/Dialect/Clift/IR/CliftTypes.h"
// keep this order
#include "revng/Model/PrimitiveType.h"

#include "revng-c/mlir/Dialect/Clift/IR/CliftAttributes.h"

#define GET_TYPEDEF_CLASSES
#include "revng-c/mlir/Dialect/Clift/IR/CliftOpsTypes.cpp.inc"

void mlir::clift::CliftDialect::registerTypes() {
  addTypes</* Include the auto-generated clift types */
#define GET_TYPEDEF_LIST
#include "revng-c/mlir/Dialect/Clift/IR/CliftOpsTypes.cpp.inc"
           /* End of types list */>();
}

/// Parse a type registered to this dialect
::mlir::Type
mlir::clift::CliftDialect::parseType(::mlir::DialectAsmParser &Parser) const {
  ::llvm::SMLoc typeLoc = Parser.getCurrentLocation();
  ::llvm::StringRef Mnemonic;
  ::mlir::Type GenType;

  auto ParseResult = generatedTypeParser(Parser, &Mnemonic, GenType);
  if (ParseResult.has_value())
    return GenType;

  Parser.emitError(typeLoc) << "unknown  type `" << Mnemonic << "` in dialect `"
                            << getNamespace() << "`";
  return {};
}

/// Print a type registered to this dialect
void mlir::clift::CliftDialect::printType(::mlir::Type Type,
                                          ::mlir::DialectAsmPrinter &Printer)
  const {

  if (::mlir::succeeded(generatedTypePrinter(Type, Printer)))
    return;
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

using LoggerType = llvm::function_ref<mlir::InFlightDiagnostic()>;
mlir::LogicalResult
mlir::clift::PrimitiveType::verify(LoggerType logger,
                                   mlir::clift::PrimitiveKind kind,
                                   unsigned long size,
                                   BoolAttr IsConst) {

  model::PrimitiveType Type(kindToKind(kind), size);
  if (not Type.verify()) {
    return logger() << "primitive type verify failed";
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

::mlir::LogicalResult
PointerType::verify(::llvm::function_ref<::mlir::InFlightDiagnostic()>
                      emitError,
                    mlir::clift::ValueType element_type,
                    uint64_t pointer_size,
                    BoolAttr IsConst) {
  if (pointer_size == 0) {
    return emitError() << "pointer type cannot have size zero";
  }
  return mlir::success();
}

::mlir::LogicalResult
ArrayType::verify(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError,
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

using namespace mlir::clift;
::mlir::LogicalResult
EnumAttr::verify(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError,
                 uint64_t ID,
                 ::llvm::StringRef name,
                 mlir::Type element_type,
                 ::llvm::ArrayRef<mlir::clift::EnumFieldAttr> Fields) {
  if (not element_type.isa<mlir::clift::PrimitiveType>()) {
    return emitError() << "type of enum must be a primitive type";
  }
  auto casted = element_type.cast<mlir::clift::PrimitiveType>();
  if (casted.getKind() != PrimitiveKind::UnsignedKind
      and casted.getKind() != PrimitiveKind::SignedKind) {
    return emitError() << "enum underlying type must be a unsigned int";
  }

  if (casted.getKind() == PrimitiveKind::UnsignedKind) {
    size_t lastValue = 0;
    for (auto field : Fields) {
      auto Max = llvm::APInt::getMaxValue(casted.getSize()).getZExtValue();
      if (field.getRawValue() > Max) {
        return emitError() << "enum field " << field.getRawValue()
                           << "is greater than the max value of the "
                              "underlying type"
                           << Max;
      }
      if (field.getRawValue() < lastValue) {
        return emitError() << "enum fields are not in ascending order";
      }
      lastValue = field.getRawValue();
    }
  }

  if (casted.getKind() == PrimitiveKind::UnsignedKind) {
    int64_t lastValue = std::numeric_limits<int64_t>::min();
    for (auto field : Fields) {
      uint64_t Raw = field.getRawValue();
      int64_t Value;
      std::memcpy(&Value, &Raw, sizeof(Value));
      auto Max = llvm::APInt::getSignedMaxValue(casted.getSize())
                   .getSExtValue();
      auto Min = llvm::APInt::getSignedMinValue(casted.getSize())
                   .getSExtValue();
      if (Value > Max) {
        return emitError() << "enum field " << Value
                           << " is greater than the max value of the "
                              "underlying type "
                           << Max;
      }
      if (Value < Min) {
        return emitError() << "enum field " << Value
                           << " is less than the min value of the "
                              "underlying type "
                           << Min;
      }
      if (Value < lastValue) {
        return emitError() << "enum fields are not in ascending order";
      }
      lastValue = Value;
    }
  }

  std::set<llvm::StringRef> Names;
  for (auto Field : Fields) {
    if (Field.getName().empty())
      continue;
    if (Names.contains(Field.getName())) {
      return emitError() << "multiple definitions of enum field attr named "
                         << Field.getName();
    }
    Names.insert(Field.getName());
  }

  return mlir::success();
}
void UnionType::walkImmediateSubElements(function_ref<void(Attribute)>
                                           walkAttrsFn,
                                         function_ref<void(Type)> walkTypesFn)
  const {
  if (not getImpl()->isInitialized())
    return;
  for (auto field : getImpl()->getFields())
    walkAttrsFn(field);
}

mlir::Attribute
UnionType::replaceImmediateSubElements(llvm::ArrayRef<mlir::Attribute>
                                         replAttrs,
                                       llvm::ArrayRef<mlir::Type> replTypes)
  const {
  revng_assert("it does not make any sense to replace the elements of a "
               "defined Union");
  return {};
}

void StructType::walkImmediateSubElements(function_ref<void(Attribute)>
                                            walkAttrsFn,
                                          function_ref<void(Type)> walkTypesFn)
  const {
  if (not getImpl()->isInitialized())
    return;
  for (auto field : getImpl()->getFields())
    walkAttrsFn(field);
}

mlir::Attribute
StructType::replaceImmediateSubElements(llvm::ArrayRef<mlir::Attribute>
                                          replAttrs,
                                        llvm::ArrayRef<mlir::Type> replTypes)
  const {
  revng_assert("it does not make any sense to replace the elements of a "
               "defined struct");
  return {};
}
