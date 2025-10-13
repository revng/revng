//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "mlir/IR/AsmState.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"

#include "revng/Clift/Clift.h"
#include "revng/Clift/CliftDialect.h"
#include "revng/Clift/CliftTypes.h"
#include "revng/Clift/ModuleVisitor.h"

#include "CliftBytecode.h"

// comment to force generated files to be included be after regular includes
#include "revng/Clift/CliftDialect.cpp.inc"

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

class CliftBytecodeDialectInterface final
  : public mlir::BytecodeDialectInterface {
public:
  CliftBytecodeDialectInterface(mlir::Dialect *Dialect) :
    BytecodeDialectInterface(Dialect) {}

  mlir::Attribute
  readAttribute(mlir::DialectBytecodeReader &Reader) const override {
    return mlir::clift::readAttr(Reader);
  }

  mlir::Type readType(mlir::DialectBytecodeReader &Reader) const override {
    return mlir::clift::readType(Reader);
  }

  mlir::LogicalResult
  writeAttribute(mlir::Attribute Attr,
                 mlir::DialectBytecodeWriter &Writer) const override {
    return mlir::clift::writeAttr(Attr, Writer);
  }

  mlir::LogicalResult
  writeType(mlir::Type Type,
            mlir::DialectBytecodeWriter &Writer) const override {
    return mlir::clift::writeType(Type, Writer);
  }
};

class ModuleVerifier : public ModuleVisitor<ModuleVerifier> {
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
    auto UnqualifiedType = mlir::cast<DefinedType>(Type.removeConst());

    auto const [Iterator, Inserted] = Definitions.try_emplace(Type.getHandle(),
                                                              UnqualifiedType);

    if (not Inserted and Iterator->second != UnqualifiedType)
      return getCurrentOp()->emitError() << "Found two distinct type "
                                            "definitions with the same "
                                            "handle: '"
                                         << UnqualifiedType.getHandle() << '\'';

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

    return mlir::success();
  }

  mlir::LogicalResult visitModuleLevelOp(mlir::Operation *Op) {
    if (not mlir::isa<GlobalOpInterface>(Op))
      return Op->emitOpError() << Op->getName()
                               << " cannot be directly nested within a"
                                  " ModuleOp.";

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

  return ModuleVerifier::visit(Module);
}

} // namespace

void CliftDialect::initialize() {
  registerTypes();
  registerOperations();
  registerAttributes();
  addInterfaces<CliftOpAsmDialectInterface, CliftBytecodeDialectInterface>();
}

mlir::LogicalResult
CliftDialect::verifyOperationAttribute(mlir::Operation *Op,
                                       mlir::NamedAttribute Attr) {
  if (Attr.getName() == getModuleAttrName())
    return verifyModuleAttr(Op, Attr.getValue());

  return mlir::success();
}

void dumpMlirOp(mlir::Operation *Op, const char *Path) {
  mlir::AsmState AsmState(Op);

  std::error_code EC;
  llvm::raw_fd_ostream OS(Path, EC);

  if (EC)
    llvm::errs() << EC.message() << "\n";
  else
    Op->print(OS, AsmState);
}

void dumpMlirModule(mlir::ModuleOp Module, const char *Path) {
  dumpMlirOp(Module, Path);
}
