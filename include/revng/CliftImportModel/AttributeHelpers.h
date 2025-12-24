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
#include "revng/Pipeline/Location.h"
#include "revng/Pipes/Ranks.h"
#include "revng/Support/Annotations.h"

namespace mlir::clift {

namespace detail {

inline llvm::SmallVector<mlir::clift::CAttributeAttr>
setOrUpdate(llvm::ArrayRef<mlir::clift::CAttributeAttr> Existing,
            mlir::clift::CAttributeAttr NewAttribute) {
  llvm::SmallVector<mlir::clift::CAttributeAttr> Result(Existing.begin(),
                                                        Existing.end());

  llvm::StringRef NewAttributeName = NewAttribute.getName().getName();

  bool AlreadyPresent = false;
  for (mlir::clift::CAttributeAttr &Attribute : Result) {
    if (Attribute.getName().getName() == NewAttributeName) {
      revng_assert(not AlreadyPresent, "Each attribute may only appear once!");
      AlreadyPresent = true;

      Attribute = NewAttribute;
    }
  }

  if (not AlreadyPresent)
    Result.emplace_back(NewAttribute);

  return Result;
}

} // namespace detail

template<ConstexprString Macro, bool IsOurs = true>
llvm::SmallVector<mlir::clift::CAttributeAttr>
setAttribute(mlir::MLIRContext *Context,
             llvm::ArrayRef<mlir::clift::CAttributeAttr>
               ExistingAttributes = {}) {
  if constexpr (IsOurs)
    ptml::Attributes.assertAttributeName<Macro>();

  using IdentifierAttr = mlir::clift::IdentifierCAttributeAttr;
  auto AttributeLocation = pipeline::location(revng::ranks::Macro,
                                              llvm::StringRef(Macro).str());
  auto AttributeName = IdentifierAttr::get(Context,
                                           Macro,
                                           AttributeLocation.toString());
  auto FullAttribute = mlir::clift::CAttributeAttr::get(Context,
                                                        AttributeName,
                                                        std::nullopt);
  return detail::setOrUpdate(ExistingAttributes, FullAttribute);
}

// Note, only single-argument versions are provided because there are currently
// no need for multi-argument attributes.

template<ConstexprString Macro, bool IsOurs = true>
llvm::SmallVector<mlir::clift::CAttributeAttr>
setAttribute(mlir::MLIRContext *Context,
             llvm::StringRef Argument,
             llvm::StringRef ArgumentLocation,
             llvm::ArrayRef<mlir::clift::CAttributeAttr>
               ExistingAttributes = {}) {
  if constexpr (IsOurs)
    ptml::Attributes.assertAnnotationName<Macro>();

  auto AttributeLocation = pipeline::location(revng::ranks::Macro,
                                              llvm::StringRef(Macro).str());

  using IdentifierAttr = mlir::clift::IdentifierCAttributeAttr;
  auto AttributeName = IdentifierAttr::get(Context,
                                           Macro,
                                           AttributeLocation.toString());
  mlir::clift::CAttributeAttrArgument
    ArgumentAttribute = IdentifierAttr::get(Context,
                                            Argument,
                                            ArgumentLocation);
  auto FullAttribute = mlir::clift::CAttributeAttr::get(Context,
                                                        AttributeName,
                                                        { ArgumentAttribute });
  return detail::setOrUpdate(ExistingAttributes, FullAttribute);
}

template<ConstexprString Macro, bool IsOurs = true>
llvm::SmallVector<mlir::clift::CAttributeAttr>
setAttribute(mlir::MLIRContext *Context,
             uint64_t Value,
             llvm::ArrayRef<mlir::clift::CAttributeAttr>
               ExistingAttributes = {}) {
  if constexpr (IsOurs)
    ptml::Attributes.assertAnnotationName<Macro>();

  revng_assert(Value == uint32_t(Value));
  llvm::APSInt LLVMValue(llvm::APInt(32, Value));

  auto AttributeLocation = pipeline::location(revng::ranks::Macro,
                                              llvm::StringRef(Macro).str());

  using IdentifierAttr = mlir::clift::IdentifierCAttributeAttr;
  auto AttributeName = IdentifierAttr::get(Context,
                                           Macro,
                                           AttributeLocation.toString());
  mlir::clift::CAttributeAttrArgument
    ArgumentAttribute = mlir::IntegerAttr::get(Context, LLVMValue);
  auto FullAttribute = mlir::clift::CAttributeAttr::get(Context,
                                                        AttributeName,
                                                        { ArgumentAttribute });
  return detail::setOrUpdate(ExistingAttributes, FullAttribute);
}

template<ConstexprString Macro, bool IsOurs = true>
llvm::SmallVector<mlir::clift::CAttributeAttr>
setAttribute(mlir::MLIRContext *Context,
             mlir::clift::ValueType Type,
             llvm::ArrayRef<mlir::clift::CAttributeAttr>
               ExistingAttributes = {}) {
  if constexpr (IsOurs)
    ptml::Attributes.assertAnnotationName<Macro>();

  auto AttributeLocation = pipeline::location(revng::ranks::Macro,
                                              llvm::StringRef(Macro).str());

  using IdentifierAttr = mlir::clift::IdentifierCAttributeAttr;
  auto AttributeName = IdentifierAttr::get(Context,
                                           Macro,
                                           AttributeLocation.toString());
  mlir::clift::CAttributeAttrArgument
    ArgumentAttribute = mlir::TypeAttr::get(Type);
  auto FullAttribute = mlir::clift::CAttributeAttr::get(Context,
                                                        AttributeName,
                                                        { ArgumentAttribute });
  return detail::setOrUpdate(ExistingAttributes, FullAttribute);
}

inline llvm::SmallVector<mlir::clift::CAttributeAttr>
fromMLIRArray(mlir::ArrayAttr CAttributes) {
  llvm::SmallVector<mlir::clift::CAttributeAttr> Result;

  for (auto Attribute : CAttributes)
    Result.emplace_back(mlir::cast<mlir::clift::CAttributeAttr>(Attribute));

  return Result;
}
inline llvm::SmallVector<mlir::clift::CAttributeAttr>
fromMLIRArray(mlir::Attribute CAttributes) {
  revng_assert(mlir::isa<mlir::ArrayAttr>(CAttributes));
  return fromMLIRArray(mlir::cast<mlir::ArrayAttr>(CAttributes));
}

inline mlir::ArrayAttr
toMLIRArray(mlir::MLIRContext *Context,
            llvm::ArrayRef<mlir::clift::CAttributeAttr> CAttributes) {
  return mlir::ArrayAttr::get(Context,
                              { CAttributes.begin(), CAttributes.end() });
}

} // namespace mlir::clift
