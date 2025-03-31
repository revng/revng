#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/ADT/SmallPtrSet.h"

#include "revng/ADT/ScopedExchange.h"
#include "revng/mlir/Dialect/Clift/IR/CliftOps.h"

namespace mlir::clift {

/// CRTP base class for ModuleOp validation. The derived class must inherit from
/// this class template publicly. This class invokes a number of customization
/// point functions on the derived object. If a customization point function is
/// not provided, its invocation is skipped. All customisation points must
/// return mlir::LogicalResult indicating the validation result.
///
/// The following example shows a derived validator definition with all
/// supported customisation points provided:
///
/// struct MyValidator : ModuleValidator<MyValidator> {
///   // Invoked for each type referenced in the module.
///   mlir::LogicalResult visitType(mlir::Type Type);
///
///   // Invoked for each attribute referenced in the module.
///   mlir::LogicalResult visitAttr(mlir::Attribute Attr);
///
///   // Invoked for each statement and expression operation nested within the
///   // current module-level operation.
///   mlir::LogicalResult visitNestedOp(mlir::Operation* Op);
///
///   // Invoked for each module level operation nested directly within the
///   // module operation.
///   mlir::LogicalResult visitModuleLevelOp(mlir::Operation* Op);
/// };
template<typename ValidatorT>
class ModuleValidator {
public:
  template<typename... Args>
    requires std::derived_from<ValidatorT, ModuleValidator>
             and std::constructible_from<ValidatorT, Args...>
  static mlir::LogicalResult validate(clift::ModuleOp Module, Args &&...args) {
    ValidatorT Validator(std::forward<Args>(args)...);
    auto &Self = static_cast<ModuleValidator &>(Validator);
    return Self.internalVisitModuleOp(Module);
  }

protected:
  ModuleValidator() = default;

  /// Always returns the operation currently being visited. This may be the
  /// module operation itself, a module-level operation, or a nested operation.
  ///
  /// It is recommended to use this operation for emitting errors.
  mlir::Operation *getCurrentOp() const { return CurrentOp; }

  /// Returns the module-level operation currently being visited, if any.
  /// Otherwise nullptr is returned.
  mlir::Operation *getCurrentModuleLevelOp() const {
    return CurrentModuleLevelOp;
  }

  /// Always returns the root module operation.
  clift::ModuleOp getCurrentModule() const { return CurrentModule; }

private:
  ValidatorT &getValidator() { return static_cast<ValidatorT &>(*this); }

  template<typename SubElementInterface>
  mlir::LogicalResult visitSubElements(SubElementInterface Interface) {
    mlir::LogicalResult R = mlir::success();

    const auto WalkType = [&](mlir::Type InnerType) {
      if (internalVisitType(InnerType).failed())
        R = mlir::failure();
    };
    const auto WalkAttr = [&](mlir::Attribute InnerAttr) {
      if (internalVisitAttr(InnerAttr).failed())
        R = mlir::failure();
    };
    Interface.walkImmediateSubElements(WalkAttr, WalkType);

    return R;
  };

  mlir::LogicalResult internalVisitType(mlir::Type Type) {
    if (not VisitedTypes.insert(Type).second)
      return mlir::success();

    if constexpr (requires { getValidator().visitType(Type); }) {
      if (getValidator().visitType(Type).failed())
        return mlir::failure();
    }

    if (auto T = mlir::dyn_cast<mlir::SubElementTypeInterface>(Type)) {
      if (visitSubElements(T).failed())
        return mlir::failure();
    }

    return mlir::success();
  }

  mlir::LogicalResult internalVisitAttr(mlir::Attribute Attr) {
    if (not VisitedAttrs.insert(Attr).second)
      return mlir::success();

    if constexpr (requires { getValidator().visitAttr(Attr); }) {
      if (getValidator().visitAttr(Attr).failed())
        return mlir::failure();
    }

    if (auto T = mlir::dyn_cast<mlir::SubElementAttrInterface>(Attr)) {
      if (visitSubElements(T).failed())
        return mlir::failure();
    }

    return mlir::success();
  }

  mlir::LogicalResult internalVisitOp(mlir::Operation *Op) {
    for (mlir::Type Type : Op->getResultTypes()) {
      if (internalVisitType(Type).failed())
        return mlir::failure();
    }

    for (mlir::Type Type : Op->getOperandTypes()) {
      if (internalVisitType(Type).failed())
        return mlir::failure();
    }

    for (auto &&Attr : Op->getAttrs()) {
      if (internalVisitAttr(Attr.getValue()).failed())
        return mlir::failure();
    }

    for (mlir::Region &Region : Op->getRegions()) {
      for (auto &Block : Region.getBlocks()) {
        for (mlir::Type ArgumentType : Block.getArgumentTypes()) {
          if (internalVisitType(ArgumentType).failed())
            return mlir::failure();
        }
      }
    }

    return mlir::success();
  }

  mlir::LogicalResult internalVisitNestedOp(mlir::Operation *Op) {
    ScopedExchange SetCurrentOp(CurrentOp, Op);

    if (internalVisitOp(Op).failed())
      return mlir::failure();

    if constexpr (requires { getValidator().visitNestedOp(Op); }) {
      if (getValidator().visitNestedOp(Op).failed())
        return mlir::failure();
    }

    return mlir::success();
  }

  mlir::LogicalResult internalVisitModuleLevelOp(mlir::Operation *Op) {
    ScopedExchange SetCurrentModuleLevelOp(CurrentModuleLevelOp, Op);
    ScopedExchange SetCurrentOp(CurrentOp, Op);

    if (internalVisitOp(Op).failed())
      return mlir::failure();

    if constexpr (requires { getValidator().visitModuleLevelOp(Op); }) {
      if (getValidator().visitModuleLevelOp(Op).failed())
        return mlir::failure();
    }

    const auto Visitor = [&](Operation *NestedOp) -> mlir::WalkResult {
      if (NestedOp == Op)
        return mlir::success();

      return internalVisitNestedOp(NestedOp);
    };

    if (Op->walk(Visitor).wasInterrupted())
      return mlir::failure();

    return mlir::success();
  }

  mlir::LogicalResult internalVisitModuleOp(clift::ModuleOp Module) {
    ScopedExchange SetCurrentModule(CurrentModule, Module);
    ScopedExchange SetCurrentOp(CurrentOp, Module.getOperation());

    if (internalVisitOp(Module.getOperation()).failed())
      return mlir::failure();

    for (mlir::Block &Block : Module.getRegion().getBlocks()) {
      for (mlir::Operation &Op : Block) {
        if (internalVisitModuleLevelOp(&Op).failed())
          return mlir::failure();
      }
    }

    if constexpr (requires { getValidator().finish(); }) {
      if (getValidator().finish().failed())
        return mlir::failure();
    }

    return mlir::success();
  }

  clift::ModuleOp CurrentModule = {};
  mlir::Operation *CurrentModuleLevelOp = nullptr;
  mlir::Operation *CurrentOp = nullptr;

  llvm::SmallPtrSet<mlir::Type, 32> VisitedTypes;
  llvm::SmallPtrSet<mlir::Attribute, 32> VisitedAttrs;
};

} // namespace mlir::clift
