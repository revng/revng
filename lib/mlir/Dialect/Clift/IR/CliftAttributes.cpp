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

#include "revng-c/mlir/Dialect/Clift/IR/Clift.h"
#include "revng-c/mlir/Dialect/Clift/IR/CliftAttributes.h"
#include "revng-c/mlir/Dialect/Clift/IR/CliftInterfaces.h"
#include "revng-c/mlir/Dialect/Clift/IR/CliftTypes.h"

#include "CliftParser.h"
#include "CliftStorage.h"

// This include should stay here for correct build procedure
//
#define GET_ATTRDEF_CLASSES
#include "revng-c/mlir/Dialect/Clift/IR/CliftAttributes.cpp.inc"

using EmitErrorType = llvm::function_ref<mlir::InFlightDiagnostic()>;

void mlir::clift::CliftDialect::registerAttributes() {
  addAttributes<StructTypeAttr, UnionTypeAttr,

  // Include the list of auto-generated attributes
#define GET_ATTRDEF_LIST
#include "revng-c/mlir/Dialect/Clift/IR/CliftAttributes.cpp.inc"
                /* End of auto-generated list */>();
}

mlir::LogicalResult mlir::clift::FieldAttr::verify(EmitErrorType EmitError,
                                                   uint64_t Offset,
                                                   mlir::Type ElementType,
                                                   llvm::StringRef Name) {
  if (auto Definition = mlir::dyn_cast<mlir::clift::DefinedType>(ElementType))
    if (mlir::isa<mlir::clift::FunctionTypeAttr>(Definition.getElementType()))
      return EmitError() << "Underlying type of field attr cannot be a "
                            "function type";
  mlir::clift::ValueType
    Casted = mlir::dyn_cast<mlir::clift::ValueType>(ElementType);
  if (Casted == nullptr) {
    return EmitError() << "Underlying type of a field attr must be a value "
                          "type";
  }
  if (Casted.getByteSize() == 0) {
    return EmitError() << "Field cannot be of zero size";
  }
  return mlir::success();
}

using EnumFieldAttr = mlir::clift::EnumFieldAttr;
mlir::LogicalResult EnumFieldAttr::verify(EmitErrorType EmitError,
                                          uint64_t RawValue,
                                          llvm::StringRef Name) {
  return mlir::success();
}

using TypedefTypeAttr = mlir::clift::TypedefTypeAttr;
mlir::LogicalResult
TypedefTypeAttr::verify(EmitErrorType EmitError,
                        uint64_t Id,
                        llvm::StringRef Name,
                        mlir::clift::ValueType UnderlyingType) {
  return mlir::success();
}

using ArgAttr = mlir::clift::FunctionArgumentAttr;
mlir::LogicalResult ArgAttr::verify(EmitErrorType EmitError,
                                    mlir::clift::ValueType underlying,
                                    llvm::StringRef Name) {
  if (underlying.getByteSize() == 0) {
    return EmitError() << "type of argument of function cannot be zero size";
  }
  return mlir::success();
}

using mlir::clift::FunctionTypeAttr;
mlir::LogicalResult
FunctionTypeAttr::verify(EmitErrorType EmitError,
                         uint64_t Id,
                         llvm::StringRef,
                         mlir::clift::ValueType ReturnType,
                         llvm::ArrayRef<mlir::clift::FunctionArgumentAttr>
                           Args) {
  if (auto Type = mlir::dyn_cast<mlir::clift::DefinedType>(ReturnType)) {
    if (mlir::isa<FunctionTypeAttr>(Type.getElementType()))
      return EmitError() << "function type cannot return another function type";
  }
  std::set<llvm::StringRef> Names;
  for (auto Arg : Args) {
    if (Arg.getName().empty())
      continue;
    if (Names.contains(Arg.getName())) {
      return EmitError() << "multiple definitions of function argument named "
                         << Arg.getName();
    }
    Names.insert(Arg.getName());
  }
  return mlir::success();
}

/// Parse a type registered to this dialect
mlir::Attribute
mlir::clift::CliftDialect::parseAttribute(mlir::DialectAsmParser &Parser,
                                          mlir::Type Type) const {
  llvm::SMLoc typeLoc = Parser.getCurrentLocation();
  llvm::StringRef Mnemonic;
  mlir::Attribute GenAttr;

  auto ParseResult = generatedAttributeParser(Parser, &Mnemonic, Type, GenAttr);
  if (ParseResult.has_value())
    return GenAttr;
  if (Mnemonic == StructTypeAttr::getMnemonic()) {
    return StructTypeAttr::parse(Parser);
  }
  if (Mnemonic == UnionTypeAttr::getMnemonic()) {
    return UnionTypeAttr::parse(Parser);
  }

  Parser.emitError(typeLoc) << "unknown  attr `" << Mnemonic << "` in dialect `"
                            << getNamespace() << "`";
  return {};
}

/// Print a type registered to this dialect
void mlir::clift::CliftDialect::printAttribute(mlir::Attribute Attr,
                                               mlir::DialectAsmPrinter &Printer)
  const {

  if (mlir::succeeded(generatedAttributePrinter(Attr, Printer)))
    return;
  if (auto Casted = Attr.dyn_cast<StructTypeAttr>()) {
    Casted.print(Printer);
    return;
  }
  if (auto Casted = Attr.dyn_cast<UnionTypeAttr>()) {
    Casted.print(Printer);
    return;
  }
  revng_abort("cannot print attribute");
}

void mlir::clift::UnionTypeAttr::print(AsmPrinter &Printer) const {
  printCompositeType(Printer, *this);
}

void mlir::clift::StructTypeAttr::print(AsmPrinter &Printer) const {
  printCompositeType(Printer, *this);
}

mlir::Attribute mlir::clift::UnionTypeAttr::parse(AsmParser &Parser) {
  return parseCompositeType<UnionTypeAttr>(Parser, /*MinSubobjects=*/1);
}

mlir::Attribute mlir::clift::StructTypeAttr::parse(AsmParser &Parser) {
  return parseCompositeType<StructTypeAttr>(Parser, /*MinSubobjects=*/0);
}

static bool isCompleteType(const mlir::Type Type) {
  if (auto T = mlir::dyn_cast<mlir::clift::DefinedType>(Type)) {
    auto Definition = T.getElementType();
    if (auto D = mlir::dyn_cast<mlir::clift::StructTypeAttr>(Definition))
      return D.isDefinition();
    if (auto D = mlir::dyn_cast<mlir::clift::UnionTypeAttr>(Definition))
      return D.isDefinition();
    return true;
  }

  if (auto T = mlir::dyn_cast<mlir::clift::ScalarTupleType>(Type))
    return T.isComplete();

  return true;
}

mlir::LogicalResult mlir::clift::StructTypeAttr::verify(EmitErrorType EmitError,
                                                        uint64_t ID) {
  return mlir::success();
}

mlir::LogicalResult
mlir::clift::StructTypeAttr::verify(const EmitErrorType EmitError,
                                    const uint64_t ID,
                                    const llvm::StringRef Name,
                                    const uint64_t Size,
                                    const llvm::ArrayRef<FieldAttr> Fields) {
  if (Size == 0)
    return EmitError() << "struct type cannot have a size of zero";

  if (not Fields.empty()) {
    uint64_t LastEndOffset = 0;

    llvm::SmallSet<llvm::StringRef, 16> NameSet;
    for (const auto &Field : Fields) {
      if (not isCompleteType(Field.getType()))
        return EmitError() << "Fields of structs must be complete types";

      if (Field.getOffset() < LastEndOffset)
        return EmitError() << "Fields of structs must be ordered by offset, "
                              "and "
                              "they cannot overlap";

      LastEndOffset = Field.getOffset()
                      + Field.getType()
                          .cast<mlir::clift::ValueType>()
                          .getByteSize();

      if (not Field.getName().empty()) {
        if (not NameSet.insert(Field.getName()).second)
          return EmitError() << "struct field names must be empty or unique";
      }
    }

    if (LastEndOffset > Size)
      return EmitError() << "offset + size of field of struct type is greater "
                            "than the struct type size.";
  }

  return mlir::success();
}

mlir::LogicalResult mlir::clift::UnionTypeAttr::verify(EmitErrorType EmitError,
                                                       uint64_t ID) {
  return mlir::success();
}

mlir::LogicalResult
mlir::clift::UnionTypeAttr::verify(EmitErrorType EmitError,
                                   uint64_t ID,
                                   llvm::StringRef,
                                   llvm::ArrayRef<FieldAttr> Fields) {
  if (Fields.empty())
    return EmitError() << "union types must have at least one field";

  llvm::SmallSet<llvm::StringRef, 16> NameSet;
  for (const auto &Field : Fields) {
    if (not isCompleteType(Field.getType()))
      return EmitError() << "Fields of unions must be complete types";

    if (Field.getOffset() != 0)
      return EmitError() << "union field offsets must be zero";

    if (not Field.getName().empty()) {
      if (not NameSet.insert(Field.getName()).second)
        return EmitError() << "union field names must be empty or unique";
    }
  }

  return mlir::success();
}

mlir::clift::StructTypeAttr
mlir::clift::StructTypeAttr::get(MLIRContext *Context, uint64_t ID) {
  return Base::get(Context, ID);
}

mlir::clift::StructTypeAttr
mlir::clift::StructTypeAttr::getChecked(EmitErrorType EmitError,
                                        MLIRContext *Context,
                                        uint64_t ID) {
  return get(Context, ID);
}

mlir::clift::StructTypeAttr
mlir::clift::StructTypeAttr::get(MLIRContext *Context,
                                 uint64_t ID,
                                 llvm::StringRef Name,
                                 uint64_t Size,
                                 llvm::ArrayRef<FieldAttr> Fields) {
  auto Result = Base::get(Context, ID);
  Result.define(Name, Size, Fields);
  return Result;
}

mlir::clift::StructTypeAttr
mlir::clift::StructTypeAttr::getChecked(EmitErrorType EmitError,
                                        MLIRContext *Context,
                                        uint64_t ID,
                                        llvm::StringRef Name,
                                        uint64_t Size,
                                        llvm::ArrayRef<FieldAttr> Fields) {
  if (failed(verify(EmitError, ID, Name, Size, Fields)))
    return {};
  return get(Context, ID, Name, Size, Fields);
}

mlir::clift::UnionTypeAttr mlir::clift::UnionTypeAttr::get(MLIRContext *Context,
                                                           uint64_t ID) {
  return Base::get(Context, ID);
}

mlir::clift::UnionTypeAttr
mlir::clift::UnionTypeAttr::getChecked(EmitErrorType EmitError,
                                       MLIRContext *Context,
                                       uint64_t ID) {
  return get(Context, ID);
}

mlir::clift::UnionTypeAttr
mlir::clift::UnionTypeAttr::get(MLIRContext *Context,
                                uint64_t ID,
                                llvm::StringRef Name,
                                llvm::ArrayRef<FieldAttr> Fields) {
  auto Result = Base::get(Context, ID);
  Result.define(Name, Fields);
  return Result;
}

mlir::clift::UnionTypeAttr
mlir::clift::UnionTypeAttr::getChecked(EmitErrorType EmitError,
                                       MLIRContext *Context,
                                       uint64_t ID,
                                       llvm::StringRef Name,
                                       llvm::ArrayRef<FieldAttr> Fields) {
  if (failed(verify(EmitError, ID, Name, Fields)))
    return {};
  return get(Context, ID, Name, Fields);
}

void mlir::clift::StructTypeAttr::define(const llvm::StringRef Name,
                                         const uint64_t Size,
                                         const llvm::ArrayRef<FieldAttr>
                                           Fields) {
  // Call into the base to mutate the type.
  LogicalResult Result = Base::mutate(Name, Fields, Size);

  // Most types expect the mutation to always succeed, but types can implement
  // custom logic for handling mutation failures.
  revng_assert(succeeded(Result)
               && "attempting to change the body of an already-initialized "
                  "type");
}

void mlir::clift::UnionTypeAttr::define(const llvm::StringRef Name,
                                        const llvm::ArrayRef<FieldAttr>
                                          Fields) {
  // Call into the base to mutate the type.
  LogicalResult Result = Base::mutate(Name, Fields);

  // Most types expect the mutation to always succeed, but types can implement
  // custom logic for handling mutation failures.
  revng_assert(succeeded(Result)
               && "attempting to change the body of an already-initialized "
                  "type");
}

uint64_t mlir::clift::StructTypeAttr::getId() const {
  return getImpl()->getID();
}

llvm::StringRef mlir::clift::StructTypeAttr::getName() const {
  return getImpl()->getName();
}

llvm::ArrayRef<mlir::clift::FieldAttr>
mlir::clift::StructTypeAttr::getFields() const {
  return getImpl()->getSubobjects();
}

bool mlir::clift::StructTypeAttr::isDefinition() const {
  return getImpl()->isInitialized();
}

uint64_t mlir::clift::StructTypeAttr::getByteSize() const {
  return getImpl()->getSize();
}

std::string mlir::clift::StructTypeAttr::getAlias() const {
  return getName().str();
}

uint64_t mlir::clift::UnionTypeAttr::getId() const {
  return getImpl()->getID();
}

llvm::StringRef mlir::clift::UnionTypeAttr::getName() const {
  return getImpl()->getName();
}

llvm::ArrayRef<mlir::clift::FieldAttr>
mlir::clift::UnionTypeAttr::getFields() const {
  return getImpl()->getSubobjects();
}

bool mlir::clift::UnionTypeAttr::isDefinition() const {
  return getImpl()->isInitialized();
}

uint64_t mlir::clift::UnionTypeAttr::getByteSize() const {
  uint64_t Max = 0;
  for (const auto &Field : getFields()) {
    mlir::Type FieldType = Field.getType();
    uint64_t Size = FieldType.cast<mlir::clift::ValueType>().getByteSize();
    Max = Size > Max ? Size : Max;
  }
  return Max;
}

std::string mlir::clift::UnionTypeAttr::getAlias() const {
  return getName().str();
}

//************************** ScalarTupleElementAttr ****************************

mlir::LogicalResult
mlir::clift::ScalarTupleElementAttr::verify(const EmitErrorType EmitError,
                                            mlir::Type Type,
                                            const llvm::StringRef Name) {
  if (not mlir::isa<mlir::clift::ValueType>(Type))
    return EmitError() << "Scalar tuple element type must be a value type";

  return mlir::success();
}
