//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"

#include "revng/mlir/Dialect/Clift/IR/Clift.h"
#include "revng/mlir/Dialect/Clift/IR/CliftOps.h"
#include "revng/mlir/Dialect/Clift/IR/CliftTypes.h"
#include "revng/mlir/Dialect/Clift/Utils/ModuleValidator.h"

// comment to force generated files to be included be after regular includes
#include "revng/mlir/Dialect/Clift/IR/CliftOpsDialect.cpp.inc"

using namespace mlir::clift;

namespace {

class CliftOpAsmDialectInterface final : public mlir::OpAsmDialectInterface {
public:
  using mlir::OpAsmDialectInterface::OpAsmDialectInterface;

  AliasResult getAlias(mlir::Attribute Attr, llvm::raw_ostream &OS) const {
    if (auto Aliasable = mlir::dyn_cast<AliasableAttr>(Attr)) {
      if (Aliasable.getAlias(OS))
        return AliasResult::FinalAlias;
    }
    return AliasResult::NoAlias;
  }

  AliasResult getAlias(mlir::Type Type, llvm::raw_ostream &OS) const {
    if (auto Aliasable = mlir::dyn_cast<AliasableType>(Type)) {
      if (Aliasable.getAlias(OS))
        return AliasResult::FinalAlias;
    }
    return AliasResult::NoAlias;
  }
};

class ModuleVerifier : public ModuleValidator<ModuleVerifier> {
  enum class LoopOrSwitch : uint8_t {
    Loop,
    Switch,
  };

public:
  // Visit a field type of a class type attribute.
  // RootType is the root class type attribute and is used to detect recursion.
  mlir::LogicalResult visitFieldType(ValueType FieldType,
                                     DefinedType RootType) {
    FieldType = dealias(FieldType);

    if (auto T = mlir::dyn_cast<DefinedType>(FieldType)) {
      if (T == RootType)
        return getCurrentOp()->emitError() << "Clift ModuleOp contains a "
                                              "recursive class type.";

      return maybeVisitClassType(T, RootType);
    }

    return mlir::success();
  }

  template<typename ClassTypeT>
  mlir::LogicalResult visitClassType(ClassTypeT Type, DefinedType RootType) {
    for (FieldAttr Field : Type.getFields()) {
      if (visitFieldType(Field.getType(), RootType).failed())
        return mlir::failure();
    }
    return mlir::success();
  }

  // Call visitClassType if Type is a class type. RootType is the root class
  // type and is used to detect recursion.
  mlir::LogicalResult maybeVisitClassType(DefinedType Type,
                                          DefinedType RootType) {
    if (auto T = mlir::dyn_cast<StructType>(Type))
      return visitClassType(T, RootType);
    if (auto T = mlir::dyn_cast<UnionType>(Type))
      return visitClassType(T, RootType);
    return mlir::success();
  }

  mlir::LogicalResult visitDefinedType(DefinedType Type) {
    auto const [Iterator, Inserted] = Definitions.try_emplace(Type.getHandle(),
                                                              Type);

    if (not Inserted and Iterator->second != Type)
      return getCurrentOp()->emitError() << "Found two distinct type "
                                            "definitions with the same "
                                            "handle: '"
                                         << Type.getHandle() << '\'';

    if (maybeVisitClassType(Type, Type).failed())
      return mlir::failure();

    return mlir::success();
  }

  mlir::LogicalResult visitValueType(ValueType Type) {
    Type = dealias(Type);

    if (not isCompleteType(Type))
      return getCurrentOp()->emitError() << "Clift ModuleOp contains an "
                                            "incomplete type";

    if (auto T = mlir::dyn_cast<DefinedType>(Type)) {
      if (visitDefinedType(T).failed())
        return mlir::failure();
    }

    return mlir::success();
  }

  mlir::LogicalResult visitType(mlir::Type Type) {
    if (Type.getDialect().getTypeID() != mlir::TypeID::get<CliftDialect>())
      return getCurrentOp()->emitError() << "Clift ModuleOp a contains "
                                            "non-Clift type";

    if (auto T = mlir::dyn_cast<ValueType>(Type)) {
      if (visitValueType(Type).failed())
        return mlir::failure();
    }

    if (auto T = mlir::dyn_cast<ClassType>(Type))
      ClassTypes.insert(T);

    return mlir::success();
  }

  mlir::LogicalResult visitNestedOp(mlir::Operation *Op) {
    if (mlir::isa<GlobalOpInterface>(Op))
      return Op->emitOpError() << Op->getName()
                               << " must be directly nested within a"
                                  " ModuleOp.";

    if (mlir::isa<SwitchBreakOp>(Op)) {
      if (not hasLoopOrSwitchParent(Op,
                                    LoopOrSwitch::Switch,
                                    /*DirectlyNested=*/true))
        return Op->emitOpError()
               << Op->getName() << " must be nested within a switch operation.";
    } else if (mlir::isa<LoopBreakOp>(Op)) {
      if (not hasLoopOrSwitchParent(Op,
                                    LoopOrSwitch::Loop,
                                    /*DirectlyNested=*/true))
        return Op->emitOpError()
               << Op->getName() << " must be nested within a loop operation.";
    } else if (mlir::isa<LoopContinueOp>(Op)) {
      if (not hasLoopOrSwitchParent(Op,
                                    LoopOrSwitch::Loop,
                                    /*DirectlyNested=*/false))
        return Op->emitOpError()
               << Op->getName() << " must be nested within a loop operation.";
    } else if (auto Sym = mlir::dyn_cast<MakeLabelOp>(Op)) {
      if (not LabelNames.insert(Sym.getName()).second)
        return Op->emitOpError()
               << Op->getName() << " conflicts with another label.";
    } else if (auto Sym = mlir::dyn_cast<LocalVariableOp>(Op)) {
      if (not LocalNames.insert(Sym.getSymName()).second)
        return Op->emitOpError()
               << Op->getName() << " conflicts with another local variable.";
    }

    return mlir::success();
  }

  mlir::LogicalResult visitModuleLevelOp(mlir::Operation *Op) {
    if (not mlir::isa<GlobalOpInterface>(Op))
      return Op->emitOpError() << Op->getName()
                               << " cannot be directly nested within a"
                                  " ModuleOp.";

    if (auto F = mlir::dyn_cast<FunctionOp>(Op)) {
      LocalNames.clear();
      LabelNames.clear();
    }

    return mlir::success();
  }

  mlir::LogicalResult finish() {
    auto EmitError = [&]() -> mlir::InFlightDiagnostic {
      return getCurrentModule()->emitError();
    };

    for (ClassType Class : ClassTypes) {
      if (auto T = mlir::dyn_cast<StructType>(Class)) {
        if (T.getDefinition().verifyDefinition(EmitError).failed())
          return mlir::failure();
      }

      if (auto T = mlir::dyn_cast<UnionType>(Class)) {
        if (T.getDefinition().verifyDefinition(EmitError).failed())
          return mlir::failure();
      }
    }
    return mlir::success();
  }

private:
  llvm::DenseMap<llvm::StringRef, DefinedType> Definitions;
  llvm::DenseSet<ClassType> ClassTypes;

  llvm::DenseSet<llvm::StringRef> LocalNames;
  llvm::DenseSet<llvm::StringRef> LabelNames;

  static std::optional<LoopOrSwitch> isLoopOrSwitch(mlir::Operation *Op) {
    if (mlir::isa<ForOp, DoWhileOp, WhileOp>(Op))
      return LoopOrSwitch::Loop;

    if (mlir::isa<SwitchOp>(Op))
      return LoopOrSwitch::Switch;

    return std::nullopt;
  }

  // Finds a loop or switch operation ancestor of the specified op. If
  // DirectlyNested is true, stops at the first such parent found, regardless of
  // its kind. Does not consider other statements, such as if-statements at all.
  bool hasLoopOrSwitchParent(mlir::Operation *Op,
                             LoopOrSwitch Kind,
                             bool DirectlyNested) {
    while (Op != getCurrentModuleLevelOp()) {
      Op = Op->getParentOp();

      if (auto OpKind = isLoopOrSwitch(Op)) {
        if (*OpKind == Kind)
          return true;

        if (DirectlyNested)
          return false;
      }
    }
    return false;
  }
};

static mlir::LogicalResult verifyModuleAttr(mlir::Operation *Op,
                                            mlir::Attribute Attr) {
  if (not mlir::isa<mlir::UnitAttr>(Attr))
    return Op->emitOpError()
           << "expected '" << CliftDialect::getModuleAttrName()
           << "' "
              "attribute to be mlir::UnitAttr.";

  auto Module = mlir::dyn_cast<mlir::ModuleOp>(Op);
  if (not Module)
    return Op->emitOpError()
           << "expected '" << CliftDialect::getModuleAttrName()
           << "' "
              "attribute to be attached to '"
           << mlir::ModuleOp::getOperationName() << "'";

  return ModuleVerifier::validate(Module);
}

} // namespace

void CliftDialect::initialize() {
  registerTypes();
  registerOperations();
  registerAttributes();
  addInterfaces<CliftOpAsmDialectInterface>();
}

mlir::LogicalResult
CliftDialect::verifyOperationAttribute(mlir::Operation *Op,
                                       mlir::NamedAttribute Attr) {
  if (Attr.getName() == getModuleAttrName())
    return verifyModuleAttr(Op, Attr.getValue());

  return mlir::success();
}
