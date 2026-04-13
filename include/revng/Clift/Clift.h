#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <bit>

#include "mlir/IR/FunctionInterfaces.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "revng/Clift/CliftAttributes.h"
#include "revng/Clift/CliftDialect.h"
#include "revng/Clift/CliftEnums.h"
#include "revng/Clift/CliftInterfaces.h"
#include "revng/Clift/CliftOpInterfaces.h"
#include "revng/Clift/CliftOpTraits.h"
#include "revng/Clift/CliftTypes.h"

namespace clift {
namespace impl {

inline constexpr llvm::StringRef LoopLabelMaskAttrName = "label_mask";

inline constexpr unsigned BreakLabelFlag = 1 << 0;
inline constexpr unsigned ContinueLabelFlag = 1 << 1;

bool verifyStatementRegion(mlir::Region &R);
bool verifyExpressionRegion(mlir::Region &R, bool Required);

unsigned getPointerArithmeticPointerOperandIndex(mlir::Operation *Op);
unsigned getPointerArithmeticOffsetOperandIndex(mlir::Operation *Op);

mlir::LogicalResult verifyUnaryIntegerMutationOp(mlir::Operation *Op);

} // namespace impl

class AttrDictView {
  mlir::DictionaryAttr Underlying;

public:
  explicit AttrDictView(mlir::DictionaryAttr Underlying) :
    Underlying(Underlying) {}

  [[nodiscard]] mlir::DictionaryAttr getDictionaryAttr() const {
    return Underlying;
  }

  [[nodiscard]] mlir::Attribute get(llvm::StringRef Name) const {
    return Underlying.get(Name);
  }

  template<std::convertible_to<mlir::Attribute> DefaultT>
  [[nodiscard]] mlir::Attribute
  get(llvm::StringRef Name, DefaultT &&Default) const {
    if (mlir::Attribute Attr = Underlying.get(Name))
      return Attr;
    return std::forward<DefaultT>(Default);
  }

  template<typename AttrT>
  [[nodiscard]] AttrT getOfType(llvm::StringRef Name) const {
    return mlir::dyn_cast_or_null<AttrT>(Underlying.get(Name));
  }

  template<typename AttrT, std::convertible_to<AttrT> DefaultT>
  [[nodiscard]] AttrT
  getOfType(llvm::StringRef Name, DefaultT &&Default) const {
    if (AttrT Attr = mlir::dyn_cast_or_null<AttrT>(Underlying.get(Name)))
      return Attr;
    return std::forward<DefaultT>(Default);
  }

  [[nodiscard]] llvm::StringRef getString(llvm::StringRef Name) const {
    auto Attr = getOfType<mlir::StringAttr>(Name);
    revng_assert(Attr);
    return Attr.getValue();
  }

  [[nodiscard]] llvm::StringRef getString(llvm::StringRef Name,
                                          llvm::StringRef Default) const {
    auto Attr = getOfType<mlir::StringAttr>(Name);
    return Attr ? Attr.getValue() : Default;
  }

  [[nodiscard]] llvm::StringRef getStringOrEmpty(llvm::StringRef Name) const {
    return getString(Name, llvm::StringRef());
  }
};

} // namespace clift

// This include should stay here for correct build procedure
#define GET_OP_CLASSES
#include "revng/Clift/Clift.h.inc"

namespace clift {

/// Returns true if the module has a Clift module attribute.
bool hasModuleAttr(mlir::ModuleOp Module);

/// Sets the Clift module attribute on the specified module.
void setModuleAttr(mlir::ModuleOp Module);

/// Returns the terminating YieldOp of the expression represented by the region,
/// or a operation if the region is not a valid expression region.
YieldOp getExpressionYieldOp(mlir::Region &R);

/// Returns the value of the expression represented by the region region, or a
/// null value if the region is not a valid expression region.
mlir::Value getExpressionValue(mlir::Region &R);

/// Returns the type of the expression represented by the region, or a null type
/// if region is not a valid expression region.
mlir::Type getExpressionType(mlir::Region &R);

} // namespace clift
