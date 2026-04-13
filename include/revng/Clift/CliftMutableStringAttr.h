#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "mlir/IR/AttributeSupport.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"

namespace clift {

class MutableStringAttrStorage;

template<typename AttrT>
using MutableStringAttrBase = mlir::Attribute::AttrBase<
  AttrT,
  mlir::Attribute,
  MutableStringAttrStorage,
  mlir::AttributeTrait::IsMutable,
  mlir::SubElementAttrInterface::Trait>;

// Pair-like attribute containing an arbitrary attribute used as key and a
// string value. Only the key participates in hashing and comparison, while the
// value can be mutated. This allows easily adding mutable string members to
// types and attributes. If the same key is used twice, mutating the string of
// one attribute will mutate the string in all attributes sharing the same key.
class MutableStringAttr : public MutableStringAttrBase<MutableStringAttr> {
protected:
  using Base = MutableStringAttrBase<MutableStringAttr>;

public:
  using Base::Base;

  static MutableStringAttr get(mlir::MLIRContext *Context, mlir::Attribute Key);

  static MutableStringAttr
  get(mlir::MLIRContext *Context, mlir::Attribute Key, llvm::StringRef Value);

  static MutableStringAttr getUnique(mlir::MLIRContext *Context,
                                     llvm::StringRef Value = {});

  mlir::Attribute getKey() const;
  llvm::StringRef getValue() const;
  void setValue(llvm::StringRef Value);

  void
  walkImmediateSubElements(llvm::function_ref<void(mlir::Attribute)> WalkAttrs,
                           llvm::function_ref<void(mlir::Type)> WalkTypes)
    const;
  mlir::Attribute
  replaceImmediateSubElements(llvm::ArrayRef<mlir::Attribute> NewAttrs,
                              llvm::ArrayRef<mlir::Type> NewTypes) const;
};

} // namespace clift
