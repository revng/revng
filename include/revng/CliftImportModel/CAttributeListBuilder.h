#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"

#include "mlir/IR/BuiltinAttributes.h"

#include "revng/ADT/ConstexprString.h"
#include "revng/Clift/Clift.h"
#include "revng/Clift/CliftAttributes.h"
#include "revng/PTML/CAttributes.h"
#include "revng/Ranks/Location.h"
#include "revng/Ranks/Ranks.h"

namespace clift {

/// This is a helper for correctly structuring c-attribute lists.
///
/// You can find examples of c-attribute lists as either `clift.c_attributes`
/// attribute attached to operations (like `clift::FunctionOp`) or
/// as an inherent attribute array (`mlir::ArrayAttr`) attached to some types,
/// like `mlir::ClassType`.
///
/// Usage is simple: you create a new object by passing the context and
/// an optional list of the existing attributes (when updating
/// `clift::FunctionOp` for example).
/// Then you just invoke `setOrUpdate` member passing the attribute name
/// (see `include/revng/PTML/CAttributes.h` for a known attribute list) as
/// a template parameter and any potential arguments as regular arguments,
/// for example:
/// ```cpp
///   MyBuilder.setOrUpdate<"_PACKED">();
///   MyBuilder.setOrUpdate<"_SIZE">(42);
///   MyBuilder.setOrUpdate<"_ABI">("$my_abi_name");
///   MyBuilder.setOrUpdate<"_UNDERLYING_TYPE">(my_clift_type);
/// ```
///
/// Note that, as the name suggests, existing attributes will be overridden!
/// For example:
/// ```cpp
///   MyBuilder.setOrUpdate<"_SIZE">(42);
///   MyBuilder.setOrUpdate<"_SIZE">(8);
/// ```
/// will result in a *single* `_SIZE` attribute with value `8`.
class CAttributeListBuilder {
public:
  using CAttributeArray = llvm::ArrayRef<clift::CAttributeAttr>;

private:
  mlir::MLIRContext *Context;
  llvm::SmallVector<clift::CAttributeAttr> Result;

public:
  explicit CAttributeListBuilder(mlir::MLIRContext *Context,
                                 CAttributeArray ExistingAttributes = {}) :
    Context(Context),
    Result(ExistingAttributes.begin(), ExistingAttributes.end()) {}

  explicit CAttributeListBuilder(mlir::MLIRContext *Context,
                                 mlir::ArrayAttr CAttributes) :
    Context(Context) {

    if (CAttributes != nullptr)
      for (auto Attribute : CAttributes)
        Result.emplace_back(mlir::cast<clift::CAttributeAttr>(Attribute));
  }

  explicit CAttributeListBuilder(mlir::MLIRContext *Context,
                                 mlir::Attribute CAttributes) :
    CAttributeListBuilder(Context,
                          mlir::cast_or_null<mlir::ArrayAttr>(CAttributes)) {}

public:
  void append(CAttributeArray ExistingAttributes) {
    Result.append(ExistingAttributes.begin(), ExistingAttributes.end());
  }

public:
  template<ConstexprString Macro>
  CAttributeListBuilder &setOrUpdate() {
    ptml::Attributes.assertAttributeName<Macro>();

    using IdentifierAttr = clift::CIdentifierAttr;
    auto AttributeLocation = pipeline::location(revng::ranks::Macro,
                                                llvm::StringRef(Macro).str());
    auto AttributeName = IdentifierAttr::get(Context,
                                             Macro,
                                             AttributeLocation.toString());
    auto FullAttribute = clift::CAttributeAttr::get(Context,
                                                    AttributeName,
                                                    nullptr);
    return setOrUpdateImpl(FullAttribute);
  }

  // Only single-argument versions are provided below because there are
  // currently no need for multi-argument attributes.

  template<ConstexprString Macro>
  CAttributeListBuilder &
  setOrUpdate(llvm::StringRef Argument, llvm::StringRef ArgumentLocation) {
    ptml::Attributes.assertAnnotationName<Macro>();

    auto AttributeLocation = pipeline::location(revng::ranks::Macro,
                                                llvm::StringRef(Macro).str());

    using IdentifierAttr = clift::CIdentifierAttr;
    auto AttributeName = IdentifierAttr::get(Context,
                                             Macro,
                                             AttributeLocation.toString());
    auto ArgAttribute = IdentifierAttr::get(Context,
                                            Argument,
                                            ArgumentLocation);
    auto Arguments = mlir::ArrayAttr::get(Context, { ArgAttribute });
    return setOrUpdateImpl(clift::CAttributeAttr::get(Context,
                                                      AttributeName,
                                                      Arguments));
  }

  template<ConstexprString Macro>
  CAttributeListBuilder &setOrUpdate(uint64_t Value) {
    ptml::Attributes.assertAnnotationName<Macro>();

    revng_assert(Value == uint32_t(Value));
    llvm::APSInt LLVMValue(llvm::APInt(32, Value));

    auto AttributeLocation = pipeline::location(revng::ranks::Macro,
                                                llvm::StringRef(Macro).str());

    using IdentifierAttr = clift::CIdentifierAttr;
    auto AttributeName = IdentifierAttr::get(Context,
                                             Macro,
                                             AttributeLocation.toString());
    auto ArgAttribute = mlir::IntegerAttr::get(Context, LLVMValue);
    auto Arguments = mlir::ArrayAttr::get(Context, { ArgAttribute });
    return setOrUpdateImpl(clift::CAttributeAttr::get(Context,
                                                      AttributeName,
                                                      Arguments));
  }

  template<ConstexprString Macro>
  CAttributeListBuilder &setOrUpdate(mlir::Type Type) {
    ptml::Attributes.assertAnnotationName<Macro>();

    auto AttributeLocation = pipeline::location(revng::ranks::Macro,
                                                llvm::StringRef(Macro).str());

    using IdentifierAttr = clift::CIdentifierAttr;
    auto AttributeName = IdentifierAttr::get(Context,
                                             Macro,
                                             AttributeLocation.toString());
    auto ArgAttribute = mlir::TypeAttr::get(Type);
    auto Arguments = mlir::ArrayAttr::get(Context, { ArgAttribute });
    return setOrUpdateImpl(clift::CAttributeAttr::get(Context,
                                                      AttributeName,
                                                      Arguments));
  }

public:
  llvm::ArrayRef<clift::CAttributeAttr> getRaw() const { return Result; }
  mlir::ArrayAttr get() const {
    return mlir::ArrayAttr::get(Context, { Result.begin(), Result.end() });
  }

private:
  CAttributeListBuilder &setOrUpdateImpl(clift::CAttributeAttr NewAttribute) {
    llvm::StringRef NewAttributeName = NewAttribute.getName().getName();

    bool AlreadyPresent = false;
    for (clift::CAttributeAttr &Attribute : Result) {
      if (Attribute.getName().getName() == NewAttributeName) {
        revng_assert(not AlreadyPresent,
                     "Each attribute may only appear once!");
        AlreadyPresent = true;

        Attribute = NewAttribute;
      }
    }

    if (not AlreadyPresent)
      Result.emplace_back(NewAttribute);

    return *this;
  }
};

} // namespace clift
