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

#include "revng/Clift/CliftEnums.h"
#include "revng/Clift/CliftInterfaces.h"
#include "revng/Clift/CliftMutableStringAttr.h"
#include "revng/Support/Assert.h"
#include "revng/Support/CDataModel.h"

namespace clift {

template<typename T>
MutableStringAttr makeNameAttr(mlir::MLIRContext *Context,
                               llvm::StringRef Handle,
                               llvm::StringRef Name = "");
template<typename T>
MutableStringAttr makeCommentAttr(mlir::MLIRContext *Context,
                                  llvm::StringRef Handle,
                                  llvm::StringRef Name = "");

} // namespace clift

// This include should stay here for correct build procedure
#define GET_ATTRDEF_CLASSES
#include "revng/Clift/CliftAttributes.h.inc"

namespace clift {

template<typename T>
MutableStringAttr makeNameAttr(mlir::MLIRContext *Context,
                               llvm::StringRef Handle,
                               llvm::StringRef Name) {
  return MutableStringAttr::get(Context,
                                StringPairAttr::get(Context,
                                                    T::NameAttrKey,
                                                    Handle),
                                Name);
}
template<typename T>
MutableStringAttr makeCommentAttr(mlir::MLIRContext *Context,
                                  llvm::StringRef Handle,
                                  llvm::StringRef Name) {
  return MutableStringAttr::get(Context,
                                StringPairAttr::get(Context,
                                                    T::CommentAttrKey,
                                                    Handle),
                                Name);
}

// VERY IMPORTANT!!!
// If you upgraded to LLVM 17 and walks on types stopped working, you need to
// read:
// discourse.llvm.org/t/custom-walk-and-replace-for-non-tablegen-types/74229
// This is very brittle and it is very likely that it will change again in
// future llvm releases
class ClassAttrStorage;

template<typename AttrT>
using ClassAttrBase = mlir::Attribute::AttrBase<
  AttrT,
  mlir::Attribute,
  ClassAttrStorage,
  mlir::AttributeTrait::IsMutable,
  TypeDefinitionAttr::Trait,
  mlir::SubElementAttrInterface::Trait>;

struct ClassDefinition {
  MutableStringAttr Name;
  MutableStringAttr Comment;

  uint64_t Size;
  llvm::ArrayRef<FieldAttr> Fields;
  llvm::ArrayRef<CAttributeAttr> CAttributes;

  ClassDefinition(MutableStringAttr Name,
                  MutableStringAttr Comment,
                  uint64_t Size,
                  llvm::ArrayRef<FieldAttr> Fields,
                  llvm::ArrayRef<CAttributeAttr> CAttributes) :
    Name(Name),
    Comment(Comment),
    Size(Size),
    Fields(Fields),
    CAttributes(CAttributes) {}

  MutableStringAttr getMutableName() const { return Name; }
  MutableStringAttr getMutableComment() const { return Comment; }

  uint64_t getSize() const { return Size; }

  llvm::ArrayRef<FieldAttr> getFields() const { return Fields; }

  llvm::ArrayRef<CAttributeAttr> getCAttributes() const { return CAttributes; }

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

  MutableStringAttr getMutableName() const {
    return getDefinition().getMutableName();
  }

  MutableStringAttr getMutableComment() const {
    return getDefinition().getMutableComment();
  }

  llvm::StringRef getName() const { return getMutableName().getValue(); }
  llvm::StringRef getComment() const { return getMutableComment().getValue(); }

  llvm::ArrayRef<FieldAttr> getFields() const {
    return getDefinition().getFields();
  }

  bool hasDefinition() const;
  const ClassDefinition *getDefinitionOrNull() const;
  const ClassDefinition &getDefinition() const;

  void
  walkImmediateSubElements(llvm::function_ref<void(mlir::Attribute)> WalkAttrs,
                           llvm::function_ref<void(mlir::Type)> WalkTypes)
    const;
  mlir::Attribute
  replaceImmediateSubElements(llvm::ArrayRef<mlir::Attribute> NewAttrs,
                              llvm::ArrayRef<mlir::Type> NewTypes) const;
};

struct StructAttr : ClassAttrImpl<StructAttr> {
  static constexpr llvm::StringRef NameAttrKey = "struct-name";
  static constexpr llvm::StringRef CommentAttrKey = "struct-comment";

  using ClassAttrImpl::ClassAttrImpl;

  static mlir::LogicalResult
  verify(llvm::function_ref<mlir::InFlightDiagnostic()> EmitError,
         llvm::StringRef Handle);

  static mlir::LogicalResult
  verify(llvm::function_ref<mlir::InFlightDiagnostic()> EmitError,
         llvm::StringRef Handle,
         const ClassDefinition &Definition);

  static mlir::LogicalResult
  verify(llvm::function_ref<mlir::InFlightDiagnostic()> EmitError,
         llvm::StringRef Handle,
         MutableStringAttr Name,
         MutableStringAttr Comment,
         uint64_t Size,
         llvm::ArrayRef<FieldAttr> Fields,
         llvm::ArrayRef<CAttributeAttr> Attributes);

  mlir::LogicalResult
  verifyDefinition(llvm::function_ref<mlir::InFlightDiagnostic()> EmitError)
    const;

  static StructAttr get(mlir::MLIRContext *Context, llvm::StringRef Handle);

  static StructAttr
  getChecked(llvm::function_ref<mlir::InFlightDiagnostic()> EmitError,
             mlir::MLIRContext *Context,
             llvm::StringRef Handle);

  static StructAttr get(mlir::MLIRContext *Context,
                        llvm::StringRef Handle,
                        const ClassDefinition &Definition);

  static StructAttr
  getChecked(llvm::function_ref<mlir::InFlightDiagnostic()> EmitError,
             mlir::MLIRContext *Context,
             llvm::StringRef Handle,
             const ClassDefinition &Definition);

  static StructAttr get(mlir::MLIRContext *Context,
                        llvm::StringRef Handle,
                        MutableStringAttr Name,
                        MutableStringAttr Comment,
                        uint64_t Size,
                        llvm::ArrayRef<FieldAttr> Fields,
                        llvm::ArrayRef<CAttributeAttr> Attributes);

  static StructAttr
  getChecked(llvm::function_ref<mlir::InFlightDiagnostic()> EmitError,
             mlir::MLIRContext *Context,
             llvm::StringRef Handle,
             MutableStringAttr Name,
             MutableStringAttr Comment,
             uint64_t Size,
             llvm::ArrayRef<FieldAttr> Fields,
             llvm::ArrayRef<CAttributeAttr> Attributes);

  uint64_t getSize() const { return getDefinition().getSize(); }

  llvm::ArrayRef<CAttributeAttr> getCAttributes() const {
    return getDefinition().getCAttributes();
  }
};

struct UnionAttr : ClassAttrImpl<UnionAttr> {
  static constexpr llvm::StringRef NameAttrKey = "union-name";
  static constexpr llvm::StringRef CommentAttrKey = "union-comment";

  using ClassAttrImpl::ClassAttrImpl;
  using ClassAttrImpl::verify;

  static mlir::LogicalResult
  verify(llvm::function_ref<mlir::InFlightDiagnostic()> EmitError,
         llvm::StringRef Handle);

  static mlir::LogicalResult
  verify(llvm::function_ref<mlir::InFlightDiagnostic()> EmitError,
         llvm::StringRef Handle,
         const ClassDefinition &Definition);

  static mlir::LogicalResult
  verify(llvm::function_ref<mlir::InFlightDiagnostic()> EmitError,
         llvm::StringRef Handle,
         MutableStringAttr Name,
         MutableStringAttr Comment,
         llvm::ArrayRef<FieldAttr> Fields,
         llvm::ArrayRef<CAttributeAttr> Attributes);

  mlir::LogicalResult
  verifyDefinition(llvm::function_ref<mlir::InFlightDiagnostic()> EmitError)
    const;

  static UnionAttr get(mlir::MLIRContext *Context, llvm::StringRef Handle);

  static UnionAttr
  getChecked(llvm::function_ref<mlir::InFlightDiagnostic()> EmitError,
             mlir::MLIRContext *Context,
             llvm::StringRef Handle);

  static UnionAttr get(mlir::MLIRContext *Context,
                       llvm::StringRef Handle,
                       const ClassDefinition &Definition);

  static UnionAttr
  getChecked(llvm::function_ref<mlir::InFlightDiagnostic()> EmitError,
             mlir::MLIRContext *Context,
             llvm::StringRef Handle,
             const ClassDefinition &Definition);

  static UnionAttr get(mlir::MLIRContext *Context,
                       llvm::StringRef Handle,
                       MutableStringAttr Name,
                       MutableStringAttr Comment,
                       llvm::ArrayRef<FieldAttr> Fields,
                       llvm::ArrayRef<CAttributeAttr> Attributes);

  static UnionAttr
  getChecked(llvm::function_ref<mlir::InFlightDiagnostic()> EmitError,
             mlir::MLIRContext *Context,
             llvm::StringRef Handle,
             MutableStringAttr Name,
             MutableStringAttr Comment,
             llvm::ArrayRef<FieldAttr> Fields,
             llvm::ArrayRef<CAttributeAttr> Attributes);

  uint64_t getSize() const;

  llvm::ArrayRef<CAttributeAttr> getCAttributes() const {
    return getDefinition().getCAttributes();
  }
};

extern template class ClassAttrImpl<StructAttr>;
extern template class ClassAttrImpl<UnionAttr>;

} // namespace clift
