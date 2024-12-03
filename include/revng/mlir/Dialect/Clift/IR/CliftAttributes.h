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
#include "revng/mlir/Dialect/Clift/IR/CliftEnums.h"
#include "revng/mlir/Dialect/Clift/IR/CliftInterfaces.h"

// This include should stay here for correct build procedure
#define GET_ATTRDEF_CLASSES
#include "revng/mlir/Dialect/Clift/IR/CliftAttributes.h.inc"

namespace mlir::clift {

// VERY IMPORTANT!!!
// If you upgraded to LLVM 17 and walks on types stopped working, you need to
// read:
// discourse.llvm.org/t/custom-walk-and-replace-for-non-tablegen-types/74229
// This is very brittle and it is very likely that it will change again in
// future llvm releases
struct StructTypeAttrStorage;
class StructTypeAttr
  : public mlir::Attribute::AttrBase<StructTypeAttr,
                                     Attribute,
                                     StructTypeAttrStorage,
                                     SubElementAttrInterface::Trait,
                                     TypeDefinitionAttr::Trait,
                                     AliasableAttr::Trait,
                                     AttributeTrait::IsMutable> {
public:
  using Base::Base;

  static StructTypeAttr get(MLIRContext *Context, uint64_t ID);

  static StructTypeAttr
  getChecked(llvm::function_ref<InFlightDiagnostic()> EmitError,
             MLIRContext *Context,
             uint64_t ID);

  static StructTypeAttr get(MLIRContext *Context,
                            uint64_t ID,
                            llvm::StringRef Name,
                            uint64_t Size,
                            llvm::ArrayRef<FieldAttr> Fields);

  static StructTypeAttr
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
  bool getAlias(llvm::raw_ostream &OS) const;

  static Attribute parse(AsmParser &Parser);
  void print(AsmPrinter &Printer) const;

  static LogicalResult verify(function_ref<InFlightDiagnostic()> EmitError,
                              uint64_t ID);
  static LogicalResult verify(function_ref<InFlightDiagnostic()> EmitError,
                              uint64_t ID,
                              llvm::StringRef Name,
                              uint64_t Size,
                              llvm::ArrayRef<FieldAttr> Fields);

  void walkImmediateSubElements(function_ref<void(Attribute)> WalkAttrs,
                                function_ref<void(Type)> WalkTypes) const;
  Attribute replaceImmediateSubElements(ArrayRef<Attribute> NewAttrs,
                                        ArrayRef<Type> NewTypes) const;
};

struct UnionTypeAttrStorage;
class UnionTypeAttr : public Attribute::AttrBase<UnionTypeAttr,
                                                 Attribute,
                                                 UnionTypeAttrStorage,
                                                 SubElementAttrInterface::Trait,
                                                 TypeDefinitionAttr::Trait,
                                                 AliasableAttr::Trait,
                                                 AttributeTrait::IsMutable> {
public:
  using Base::Base;

  static UnionTypeAttr get(MLIRContext *Context, uint64_t ID);

  static UnionTypeAttr
  getChecked(llvm::function_ref<InFlightDiagnostic()> EmitError,
             MLIRContext *Context,
             uint64_t ID);

  static UnionTypeAttr get(MLIRContext *Context,
                           uint64_t ID,
                           llvm::StringRef Name,
                           llvm::ArrayRef<FieldAttr> Fields);

  static UnionTypeAttr
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
  bool getAlias(llvm::raw_ostream &OS) const;

  static Attribute parse(AsmParser &Parser);
  void print(AsmPrinter &Printer) const;

  static LogicalResult verify(function_ref<InFlightDiagnostic()> EmitError,
                              uint64_t ID);
  static LogicalResult verify(function_ref<InFlightDiagnostic()> EmitError,
                              uint64_t ID,
                              llvm::StringRef Name,
                              llvm::ArrayRef<FieldAttr> Fields);

  // since mlir types and attributes are immutable, the infrastructure must
  // provide to replace a subelement of the hierarchy. These methods allow
  // to do. Notice that since LLVM17 these are no longer methods requested
  // by the SubElementAttrInterface but are instead a builtin property of
  // all types and attributes, so it will break.
  void walkImmediateSubElements(function_ref<void(Attribute)> WalkAttrs,
                                function_ref<void(Type)> WalkTypes) const;
  Attribute replaceImmediateSubElements(ArrayRef<Attribute> NewAttrs,
                                        ArrayRef<Type> NewTypes) const;
};
} // namespace mlir::clift
