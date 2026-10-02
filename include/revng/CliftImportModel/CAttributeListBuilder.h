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
/// You can find examples of c-attribute lists as either
/// `clift.c_attribute_list` attribute attached to operations (such as
/// `clift::FunctionOp`) or types (such as `clift::StructType`).
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
    Context(Context), Result(ExistingAttributes) {}

  explicit CAttributeListBuilder(mlir::MLIRContext *Context,
                                 CAttributeListAttr AttributeList) :
    Context(Context) {

    if (AttributeList)
      Result.assign(AttributeList.begin(), AttributeList.end());
  }

public:
  void append(CAttributeListAttr AttributeList) {
    if (AttributeList)
      Result.append(AttributeList.begin(), AttributeList.end());
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
  CAttributeListBuilder &setOrUpdate(llvm::APSInt Value) {
    revng_assert(Value.getBitWidth() <= 64,
                 "Integers wider than 64 bits are not representable in C.");

    ptml::Attributes.assertAnnotationName<Macro>();
    auto AttributeLocation = pipeline::location(revng::ranks::Macro,
                                                llvm::StringRef(Macro).str());

    using IdentifierAttr = clift::CIdentifierAttr;
    auto AttributeName = IdentifierAttr::get(Context,
                                             Macro,
                                             AttributeLocation.toString());
    auto ArgAttribute = mlir::IntegerAttr::get(Context, Value);
    auto Arguments = mlir::ArrayAttr::get(Context, { ArgAttribute });
    return setOrUpdateImpl(clift::CAttributeAttr::get(Context,
                                                      AttributeName,
                                                      Arguments));
  }

  template<ConstexprString Macro, std::integral IntegerT>
  CAttributeListBuilder &setOrUpdate(IntegerT Integer) {
    auto Value = llvm::APSInt(llvm::APInt(64, static_cast<uint64_t>(Integer)),
                              std::is_unsigned_v<IntegerT>);

    return setOrUpdate<Macro>(std::move(Value));
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
  [[nodiscard]] bool empty() const { return Result.empty(); }

  [[nodiscard]] CAttributeListAttr getAttributeList() const {
    revng_assert(not Result.empty());
    return CAttributeListAttr::get(Context, Result);
  }

  [[nodiscard]] CAttributeListAttr getAttributeListOrNull() const {
    return Result.empty() ? CAttributeListAttr(nullptr) : getAttributeList();
  }

  [[nodiscard]] CAttributeArray::iterator begin() const {
    return Result.begin();
  }

  [[nodiscard]] CAttributeArray::iterator end() const { return Result.end(); }

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
