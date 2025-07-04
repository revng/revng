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
class ClassAttrStorage;

template<typename AttrT>
using ClassAttrBase = Attribute::AttrBase<AttrT,
                                          Attribute,
                                          ClassAttrStorage,
                                          AttributeTrait::IsMutable,
                                          SubElementAttrInterface::Trait>;

struct ClassDefinition {
  llvm::StringRef Name;
  uint64_t Size;
  llvm::ArrayRef<FieldAttr> Fields;

  llvm::StringRef getName() const { return Name; }

  uint64_t getSize() const { return Size; }

  llvm::ArrayRef<FieldAttr> getFields() const { return Fields; }

  friend bool operator==(const ClassDefinition &,
                         const ClassDefinition &) = default;
};

template<typename AttrT>
class ClassAttrImpl : public ClassAttrBase<AttrT> {
protected:
  using Base = ClassAttrBase<AttrT>;

public:
  using Base::Base;

  static AttrT get(mlir::MLIRContext *Context, llvm::StringRef Handle);

  llvm::StringRef getHandle() const;

  llvm::StringRef getName() const { return getDefinition().getName(); }

  llvm::ArrayRef<FieldAttr> getFields() const {
    return getDefinition().getFields();
  }

  bool hasDefinition() const;
  const ClassDefinition *getDefinitionOrNull() const;
  const ClassDefinition &getDefinition() const;

  void walkImmediateSubElements(llvm::function_ref<void(Attribute)> WalkAttrs,
                                llvm::function_ref<void(Type)> WalkTypes) const;
  Attribute replaceImmediateSubElements(llvm::ArrayRef<Attribute> NewAttrs,
                                        llvm::ArrayRef<Type> NewTypes) const;
};

struct StructAttr : ClassAttrImpl<StructAttr> {
  using ClassAttrImpl::ClassAttrImpl;

  static LogicalResult
  verify(llvm::function_ref<InFlightDiagnostic()> EmitError,
         llvm::StringRef Handle);

  static mlir::LogicalResult
  verify(llvm::function_ref<InFlightDiagnostic()> EmitError,
         llvm::StringRef Handle,
         const ClassDefinition &Definition);

  static mlir::LogicalResult
  verify(llvm::function_ref<InFlightDiagnostic()> EmitError,
         llvm::StringRef Handle,
         llvm::StringRef Name,
         uint64_t Size,
         llvm::ArrayRef<FieldAttr> Fields);

  mlir::LogicalResult
  verifyDefinition(llvm::function_ref<InFlightDiagnostic()> EmitError) const;

  static StructAttr get(MLIRContext *Context, llvm::StringRef Handle);

  static StructAttr
  getChecked(llvm::function_ref<InFlightDiagnostic()> EmitError,
             MLIRContext *Context,
             llvm::StringRef Handle);

  static StructAttr get(MLIRContext *Context,
                        llvm::StringRef Handle,
                        const ClassDefinition &Definition);

  static StructAttr
  getChecked(llvm::function_ref<InFlightDiagnostic()> EmitError,
             MLIRContext *Context,
             llvm::StringRef Handle,
             const ClassDefinition &Definition);

  static StructAttr get(MLIRContext *Context,
                        llvm::StringRef Handle,
                        llvm::StringRef Name,
                        uint64_t Size,
                        llvm::ArrayRef<FieldAttr> Fields);

  static StructAttr
  getChecked(llvm::function_ref<InFlightDiagnostic()> EmitError,
             MLIRContext *Context,
             llvm::StringRef Handle,
             llvm::StringRef Name,
             uint64_t Size,
             llvm::ArrayRef<FieldAttr> Fields);

  uint64_t getSize() const { return getDefinition().getSize(); }
};

struct UnionAttr : ClassAttrImpl<UnionAttr> {
  using ClassAttrImpl::ClassAttrImpl;
  using ClassAttrImpl::verify;

  static LogicalResult
  verify(llvm::function_ref<InFlightDiagnostic()> EmitError,
         llvm::StringRef Handle);

  static LogicalResult
  verify(llvm::function_ref<InFlightDiagnostic()> EmitError,
         llvm::StringRef Handle,
         const ClassDefinition &Definition);

  static LogicalResult
  verify(llvm::function_ref<InFlightDiagnostic()> EmitError,
         llvm::StringRef Handle,
         llvm::StringRef Name,
         llvm::ArrayRef<FieldAttr> Fields);

  mlir::LogicalResult
  verifyDefinition(llvm::function_ref<InFlightDiagnostic()> EmitError) const;

  static UnionAttr get(MLIRContext *Context, llvm::StringRef Handle);

  static UnionAttr
  getChecked(llvm::function_ref<InFlightDiagnostic()> EmitError,
             MLIRContext *Context,
             llvm::StringRef Handle);

  static UnionAttr get(MLIRContext *Context,
                       llvm::StringRef Handle,
                       const ClassDefinition &Definition);

  static UnionAttr
  getChecked(llvm::function_ref<InFlightDiagnostic()> EmitError,
             MLIRContext *Context,
             llvm::StringRef Handle,
             const ClassDefinition &Definition);

  static UnionAttr get(MLIRContext *Context,
                       llvm::StringRef Handle,
                       llvm::StringRef Name,
                       llvm::ArrayRef<FieldAttr> Fields);

  static UnionAttr
  getChecked(llvm::function_ref<InFlightDiagnostic()> EmitError,
             MLIRContext *Context,
             llvm::StringRef Handle,
             llvm::StringRef Name,
             llvm::ArrayRef<FieldAttr> Fields);

  uint64_t getSize() const;
};

extern template class ClassAttrImpl<StructAttr>;
extern template class ClassAttrImpl<UnionAttr>;

} // namespace mlir::clift
