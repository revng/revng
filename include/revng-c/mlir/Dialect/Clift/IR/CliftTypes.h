#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/ADT/TypeSwitch.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/TypeSupport.h"
#include "mlir/IR/Types.h"

#include "revng/Support/Assert.h"

#include "revng-c/mlir/Dialect/Clift/IR/Clift.h"
#include "revng-c/mlir/Dialect/Clift/IR/CliftAttributes.h"
#include "revng-c/mlir/Dialect/Clift/IR/CliftInterfaces.h"

// This include should stay here for correct build procedure
#define GET_TYPEDEF_CLASSES
#include "revng-c/mlir/Dialect/Clift/IR/CliftOpsTypes.h.inc"

namespace mlir::clift {

struct ScalarTupleTypeStorage;

/// Scalar tuples are used to represent the multiple register return values of
/// raw function types. They are much like structs, but permit only scalar
/// element types and have no explicit offsets for their elements.
///
/// The scalar tuple type is intended as a temporary solution to be replaced in
/// the future by auto-generated struct types.
class ScalarTupleType : public Type::TypeBase<ScalarTupleType,
                                              Type,
                                              ScalarTupleTypeStorage,
                                              ValueType::Trait,
                                              SubElementTypeInterface::Trait,
                                              AliasableType::Trait,
                                              TypeTrait::IsMutable> {
public:
  static constexpr llvm::StringLiteral getMnemonic() { return { "tuple" }; }

  using Base::Base;

  static LogicalResult verify(function_ref<InFlightDiagnostic()> EmitError,
                              uint64_t ID);

  static LogicalResult verify(function_ref<InFlightDiagnostic()> EmitError,
                              uint64_t ID,
                              llvm::StringRef Name,
                              llvm::ArrayRef<ScalarTupleElementAttr> Elements);

  static ScalarTupleType get(MLIRContext *Context, uint64_t ID);

  static ScalarTupleType
  getChecked(llvm::function_ref<InFlightDiagnostic()> EmitError,
             MLIRContext *Context,
             uint64_t ID);

  static ScalarTupleType get(MLIRContext *Context,
                             uint64_t ID,
                             llvm::StringRef Name,
                             llvm::ArrayRef<ScalarTupleElementAttr> Elements);

  static ScalarTupleType
  getChecked(llvm::function_ref<InFlightDiagnostic()> EmitError,
             MLIRContext *Context,
             uint64_t ID,
             llvm::StringRef Name,
             llvm::ArrayRef<ScalarTupleElementAttr> Elements);

  void define(llvm::StringRef Name,
              llvm::ArrayRef<ScalarTupleElementAttr> Elements);

  [[nodiscard]] uint64_t getId() const;
  [[nodiscard]] llvm::StringRef getName() const;
  [[nodiscard]] llvm::ArrayRef<ScalarTupleElementAttr> getElements() const;

  [[nodiscard]] bool isComplete() const;
  [[nodiscard]] uint64_t getByteSize() const;
  [[nodiscard]] bool getAlias(llvm::raw_ostream &OS) const;
  [[nodiscard]] BoolAttr getIsConst() const;

  // ScalarTupleType does not support const. That is because it is only used as
  // a return type, and const return types do not make sense.
  [[nodiscard]] clift::ValueType addConst() const { return *this; }
  [[nodiscard]] clift::ValueType removeConst() const { return *this; }

  static Type parse(AsmParser &Parser);
  void print(AsmPrinter &Printer) const;

  void walkImmediateSubElements(function_ref<void(Attribute)> WalkAttr,
                                function_ref<void(Type)> WalkType) const;
  Type replaceImmediateSubElements(ArrayRef<Attribute> NewAttrs,
                                   ArrayRef<Type> NewTypes) const;
};

} // namespace mlir::clift
