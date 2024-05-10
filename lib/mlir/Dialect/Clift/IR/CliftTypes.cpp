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
#include "CliftTypeHelpers.h"

#define GET_TYPEDEF_CLASSES
#include "revng-c/mlir/Dialect/Clift/IR/CliftOpsTypes.cpp.inc"

using namespace mlir::clift;

using EmitErrorType = llvm::function_ref<mlir::InFlightDiagnostic()>;

//===---------------------------- CliftDialect ----------------------------===//

void CliftDialect::registerTypes() {
  addTypes<ScalarTupleType, /* Include the auto-generated clift types */
#define GET_TYPEDEF_LIST
#include "revng-c/mlir/Dialect/Clift/IR/CliftOpsTypes.cpp.inc"
           /* End of auto-generated list */>();
}

/// Parse a type registered to this dialect
mlir::Type CliftDialect::parseType(mlir::DialectAsmParser &Parser) const {
  const llvm::SMLoc TypeLoc = Parser.getCurrentLocation();

  llvm::StringRef Mnemonic;
  if (mlir::Type GenType;
      generatedTypeParser(Parser, &Mnemonic, GenType).has_value())
    return GenType;

  if (Mnemonic == ScalarTupleType::getMnemonic())
    return ScalarTupleType::parse(Parser);

  Parser.emitError(TypeLoc) << "unknown type `" << Mnemonic << "` in dialect `"
                            << getNamespace() << "`";
  return {};
}

/// Print a type registered to this dialect
void CliftDialect::printType(mlir::Type Type,
                             mlir::DialectAsmPrinter &Printer) const {
  if (mlir::succeeded(generatedTypePrinter(Type, Printer)))
    return;
  if (auto T = mlir::dyn_cast<ScalarTupleType>(Type)) {
    T.print(Printer);
    return;
  }
  revng_abort("cannot print type");
}

//===---------------------------- PrimitiveType ---------------------------===//

static constexpr model::PrimitiveTypeKind::Values
kindToKind(const PrimitiveKind Kind) {
  return static_cast<model::PrimitiveTypeKind::Values>(Kind);
}

/// Test that kindToKind converts each clift::PrimitiveKind to the matching
/// model::PrimitiveTypeKind. Use a switch converting in the opposite direction
/// in order to produce a warning if a new primitive kind is ever added.
static consteval bool testKindToKind() {
  PrimitiveKind UninitializedKind;
  const auto TestSwitch = [&](const model::PrimitiveTypeKind::Values Kind) {
    switch (Kind) {
    case model::PrimitiveTypeKind::Float:
      return PrimitiveKind::FloatKind;
    case model::PrimitiveTypeKind::Generic:
      return PrimitiveKind::GenericKind;
    case model::PrimitiveTypeKind::Number:
      return PrimitiveKind::NumberKind;
    case model::PrimitiveTypeKind::PointerOrNumber:
      return PrimitiveKind::PointerOrNumberKind;
    case model::PrimitiveTypeKind::Signed:
      return PrimitiveKind::SignedKind;
    case model::PrimitiveTypeKind::Unsigned:
      return PrimitiveKind::UnsignedKind;
    case model::PrimitiveTypeKind::Void:
      return PrimitiveKind::VoidKind;

    case model::PrimitiveTypeKind::Invalid:
    case model::PrimitiveTypeKind::Count:
      // Unreachable. This causes an error during constant evaluation.
      return UninitializedKind;
    }
  };

  for (int I = 0; I < static_cast<int>(model::PrimitiveTypeKind::Count); ++I) {
    auto const Kind = static_cast<model::PrimitiveTypeKind::Values>(I);
    if (Kind != model::PrimitiveTypeKind::Invalid) {
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
                                          BoolAttr IsConst) {
  model::PrimitiveType Type(kindToKind(Kind), Size);
  if (not Type.verify()) {
    return EmitError() << "primitive type verify failed";
  }
  return mlir::success();
}

std::string PrimitiveType::getAlias() const {
  model::PrimitiveType Type(kindToKind(getKind()), getByteSize());
  return isConst() ? std::string("const_") + serializeToString(Type.name()) :
                     serializeToString(Type.name());
}

//===----------------------------- PointerType ----------------------------===//

mlir::LogicalResult PointerType::verify(EmitErrorType EmitError,
                                        ValueType ElementType,
                                        uint64_t PointerSize,
                                        BoolAttr IsConst) {
  switch (PointerSize) {
  case 4:
  case 8:
    break;
  default:
    return EmitError() << "invalid pointer size: " << PointerSize;
  }
  return mlir::success();
}

uint64_t PointerType::getByteSize() const {
  return getPointerSize();
}

//===------------------------------ ArrayType -----------------------------===//

mlir::LogicalResult ArrayType::verify(EmitErrorType EmitError,
                                      ValueType ElementType,
                                      uint64_t ElementCount,
                                      BoolAttr IsConst) {
  if (not isObjectType(ElementType))
    return EmitError() << "Array type element type must be an object type.";
  if (ElementCount == 0)
    return EmitError() << "Array type must have no less than one element.";

  return mlir::success();
}

uint64_t ArrayType::getByteSize() const {
  return getElementType().getByteSize() * getElementsCount();
}

//===----------------------------- DefinedType ----------------------------===//

mlir::LogicalResult DefinedType::verify(EmitErrorType EmitError,
                                        TypeDefinitionAttr Definition,
                                        BoolAttr IsConst) {
  return mlir::success();
}

uint64_t DefinedType::id() {
  return getElementType().id();
}

llvm::StringRef DefinedType::name() {
  return getElementType().name();
}

uint64_t DefinedType::getByteSize() const {
  return mlir::cast<SizedType>(getElementType()).getByteSize();
}

std::string DefinedType::getAlias() const {
  if (auto Casted = getElementType().dyn_cast<AliasableAttr>()) {
    if (Casted.getAlias().empty())
      return "";
    return isConst() ? "const_" + Casted.getAlias() : Casted.getAlias();
  }
  return "";
}

//===--------------------------- ScalarTupleType --------------------------===//

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
    Size += Element.getType().getByteSize();
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
