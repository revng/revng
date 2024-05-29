#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "mlir/IR/AttributeSupport.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/DialectImplementation.h"

#include "revng/Support/Assert.h"

#include "revng-c/mlir/Dialect/Clift/IR/CliftEnums.h"
#include "revng-c/mlir/Dialect/Clift/IR/CliftInterfaces.h"

// This include should stay here for correct build procedure
#define GET_ATTRDEF_CLASSES
#include "revng-c/mlir/Dialect/Clift/IR/CliftAttributes.h.inc"

namespace mlir::clift {

// VERY IMPORTANT!!!
// If you upgraded to LLVM 17 and walks on types stopped working, you need to
// read:
// discourse.llvm.org/t/custom-walk-and-replace-for-non-tablegen-types/74229
// This is very brittle and it is very likely that it will change again in
// future llvm releases
struct StructTypeStorage;
class StructType
  : public ::mlir::Attribute::AttrBase<StructType,
                                       Attribute,
                                       StructTypeStorage,
                                       SubElementAttrInterface::Trait,
                                       SizedType::Trait,
                                       TypeDefinition::Trait,
                                       AliasableAttr::Trait,
                                       AttributeTrait::IsMutable> {
public:
  using Base::Base;

  static StructType get(MLIRContext *Context, uint64_t ID);

  static StructType
  getChecked(llvm::function_ref<InFlightDiagnostic()> EmitError,
             MLIRContext *Context,
             uint64_t ID);

  static StructType get(MLIRContext *Context,
                        uint64_t ID,
                        llvm::StringRef Name,
                        uint64_t Size,
                        llvm::ArrayRef<FieldAttr> Fields);

  static StructType
  getChecked(llvm::function_ref<InFlightDiagnostic()> EmitError,
             MLIRContext *Context,
             uint64_t ID,
             llvm::StringRef Name,
             uint64_t Size,
             llvm::ArrayRef<FieldAttr> Fields);

  static llvm::StringLiteral getMnemonic() { return { "struct" }; }

  void
  define(llvm::StringRef Name, uint64_t Size, llvm::ArrayRef<FieldAttr> Fields);

  uint64_t getId() const;
  llvm::StringRef getName() const;
  llvm::ArrayRef<FieldAttr> getFields() const;

  bool isDefinition() const;
  uint64_t getByteSize() const;
  std::string getAlias() const;

  static Attribute parse(AsmParser &Parser);
  void print(AsmPrinter &Printer) const;

  static LogicalResult verify(function_ref<InFlightDiagnostic()> emitError,
                              uint64_t ID);
  static LogicalResult verify(function_ref<InFlightDiagnostic()> emitError,
                              uint64_t ID,
                              llvm::StringRef Name,
                              uint64_t Size,
                              llvm::ArrayRef<FieldAttr> Fields);

  void walkImmediateSubElements(function_ref<void(Attribute)> walkAttrsFn,
                                function_ref<void(Type)> walkTypesFn) const;
  Attribute replaceImmediateSubElements(ArrayRef<Attribute> replAttrs,
                                        ArrayRef<Type> replTypes) const;
};

struct UnionTypeStorage;
class UnionType : public Attribute::AttrBase<UnionType,
                                             Attribute,
                                             UnionTypeStorage,
                                             SubElementAttrInterface::Trait,
                                             SizedType::Trait,
                                             TypeDefinition::Trait,
                                             AliasableAttr::Trait,
                                             AttributeTrait::IsMutable> {
public:
  using Base::Base;

  static UnionType get(MLIRContext *Context, uint64_t ID);

  static UnionType
  getChecked(llvm::function_ref<InFlightDiagnostic()> EmitError,
             MLIRContext *Context,
             uint64_t ID);

  static UnionType get(MLIRContext *Context,
                       uint64_t ID,
                       llvm::StringRef Name,
                       llvm::ArrayRef<FieldAttr> Fields);

  static UnionType
  getChecked(llvm::function_ref<InFlightDiagnostic()> EmitError,
             MLIRContext *Context,
             uint64_t ID,
             llvm::StringRef Name,
             llvm::ArrayRef<FieldAttr> Fields);

  static llvm::StringLiteral getMnemonic() { return { "union" }; }

  void define(llvm::StringRef Name, llvm::ArrayRef<FieldAttr> Fields);

  uint64_t getId() const;
  llvm::StringRef getName() const;
  llvm::ArrayRef<FieldAttr> getFields() const;

  bool isDefinition() const;
  uint64_t getByteSize() const;
  std::string getAlias() const;

  static Attribute parse(AsmParser &Parser);
  void print(AsmPrinter &Printer) const;

  static LogicalResult verify(function_ref<InFlightDiagnostic()> emitError,
                              uint64_t ID);
  static LogicalResult verify(function_ref<InFlightDiagnostic()> emitError,
                              uint64_t ID,
                              llvm::StringRef Name,
                              llvm::ArrayRef<FieldAttr> Fields);

  // since mlir types and attributes are immutable, the infrastructure must
  // provide to replace a subelement of the hierarchy. These methods allow
  // to do. Notice that since LLVM17 these are no longer methods requested
  // by the SubElementAttrInterface but are instead a builtin property of
  // all types and attributes, so it will break.
  void walkImmediateSubElements(function_ref<void(Attribute)> walkAttrsFn,
                                function_ref<void(Type)> walkTypesFn) const;
  Attribute replaceImmediateSubElements(ArrayRef<Attribute> replAttrs,
                                        ArrayRef<Type> replTypes) const;
};
} // namespace mlir::clift
