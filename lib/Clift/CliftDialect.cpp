//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <optional>

#include "llvm/ADT/SmallPtrSet.h"

#include "mlir/IR/AsmState.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"

#include "revng/ADT/RecursiveCoroutine.h"
#include "revng/Clift/Clift.h"
#include "revng/Clift/CliftDialect.h"
#include "revng/Clift/CliftTypes.h"
#include "revng/Clift/ModuleVisitor.h"

#include "CliftBytecode.h"

// comment to force generated files to be included be after regular includes
#include "revng/Clift/CliftDialect.cpp.inc"

using namespace clift;

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
    return clift::readAttr(Reader);
  }

  mlir::Type readType(mlir::DialectBytecodeReader &Reader) const override {
    return clift::readType(Reader);
  }

  mlir::LogicalResult
  writeAttribute(mlir::Attribute Attr,
                 mlir::DialectBytecodeWriter &Writer) const override {
    return clift::writeAttr(Attr, Writer);
  }

  mlir::LogicalResult
  writeType(mlir::Type Type,
            mlir::DialectBytecodeWriter &Writer) const override {
    return clift::writeType(Type, Writer);
  }
};

/// Answers whether a class type contains itself through a chain of fields
///
/// A class type is walked once for as long as one of these lives: lying on a
/// cycle is a property of the type graph alone and not of where a walk began,
/// so what one walk settles holds for every later one. Without that, a class
/// type is walked once per class type that can reach it, which is exponential
/// in the depth of the type graph.
class RecursionDetector {
private:
  /// The class types on the chain of fields currently being walked
  llvm::SmallPtrSet<mlir::Type, 16> OnPath;

  /// The class types already shown not to contain themselves
  llvm::SmallPtrSet<mlir::Type, 16> Acyclic;

public:
  /// \return true if \p Type contains itself through a chain of fields
  ///
  /// Coming back to a class type that is still on the chain being walked is
  /// what says so, which also finds a cycle that \p Type is not part of.
  RecursiveCoroutine<bool> isRecursive(ClassType Type) {
    mlir::Type Key = removeConst(mlir::Type(Type));

    if (Acyclic.contains(Key))
      rc_return false;

    if (not OnPath.insert(Key).second)
      rc_return true;

    bool Recursive = rc_recur hasRecursiveField(Type);
    if (not Recursive)
      Acyclic.insert(Key);

    OnPath.erase(Key);
    rc_return Recursive;
  }

private:
  /// \return true if \p Type has a recursive field type.
  RecursiveCoroutine<bool> hasRecursiveField(ClassType Type) {
    for (FieldAttr Field : Type.getFields()) {
      if (rc_recur isRecursiveField(Field.getType()))
        rc_return true;
    }
    rc_return false;
  }

  /// \return true if \p FieldType is a class type that contains itself
  RecursiveCoroutine<bool> isRecursiveField(mlir::Type FieldType) {
    while (true) {
      FieldType = unwrapTypedefs(FieldType);
      auto Array = mlir::dyn_cast<ArrayType>(FieldType);
      if (not Array)
        break;
      FieldType = Array.getElementType();
    }

    if (auto T = mlir::dyn_cast<ClassType>(FieldType))
      rc_return rc_recur isRecursive(T);

    rc_return false;
  }
};

class ModuleVerifier : public ModuleVisitor<ModuleVerifier> {
  enum class LoopOrSwitch : uint8_t {
    Loop,
    Switch,
  };

public:
  mlir::LogicalResult visitDefinedType(DefinedType Type) {
    auto UnqualifiedType = mlir::cast<DefinedType>(removeConst(Type));

    auto const [Iterator, Inserted] = Definitions.try_emplace(Type.getHandle(),
                                                              UnqualifiedType);

    if (not Inserted and Iterator->second != UnqualifiedType)
      return getCurrentOp()->emitError() << "Found two distinct type "
                                            "definitions with the same "
                                            "handle: '"
                                         << UnqualifiedType.getHandle() << '\'';

    if (auto Class = mlir::dyn_cast<ClassType>(Type)) {
      if (Recursion.isRecursive(Class))
        return getCurrentOp()->emitError() << "Clift ModuleOp contains a "
                                              "recursive class type.";

      ClassTypes.insert(Class);
    }

    return mlir::success();
  }

  mlir::LogicalResult visitAddressableType(AddressableType Type) {
    Type = unwrapTypedefs(Type);

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

    if (auto T = mlir::dyn_cast<AddressableType>(Type)) {
      if (visitAddressableType(Type).failed())
        return mlir::failure();
    }

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
  RecursionDetector Recursion;
};

static mlir::LogicalResult verifyModuleAttr(mlir::Operation *Op,
                                            mlir::Attribute Attr) {
  if (not mlir::isa<mlir::UnitAttr>(Attr))
    return Op->emitOpError()
           << "expected '" << CliftDialect::getModuleAttrName()
           << "' attribute to be mlir::UnitAttr.";

  auto Module = mlir::dyn_cast<mlir::ModuleOp>(Op);
  if (not Module)
    return Op->emitOpError()
           << "expected '" << CliftDialect::getModuleAttrName()
           << "' attribute to be attached to '"
           << mlir::ModuleOp::getOperationName() << "'";

  return ModuleVerifier::visit(Module);
}

static mlir::LogicalResult verifyDataModelAttr(mlir::Operation *Op,
                                               mlir::Attribute Attr) {
  if (not mlir::isa<DataModelAttr>(Attr))
    return Op->emitOpError()
           << "expected '" << CliftDialect::getDataModelAttrName()
           << "' attribute to be of type DataModelAttr.";

  if (not mlir::isa<mlir::ModuleOp>(Op))
    return Op->emitOpError()
           << "expected '" << CliftDialect::getDataModelAttrName()
           << "' attribute to be attached to '"
           << mlir::ModuleOp::getOperationName() << "'";

  return mlir::success();
}

} // namespace

struct clift::CliftDialectImpl {
  std::optional<CDataModel> DefaultDataModel;
};

void CliftDialect::initialize() {
  Impl = new CliftDialectImpl();

  registerTypes();
  registerOperations();
  registerAttributes();
  addInterfaces<CliftOpAsmDialectInterface, CliftBytecodeDialectInterface>();
}

CliftDialect::~CliftDialect() {
  delete Impl;
}

const CDataModel *CliftDialect::getDefaultDataModel() const {
  if (const auto &DM = Impl->DefaultDataModel)
    return &*DM;

  return nullptr;
}

void CliftDialect::setDefaultDataModel(const CDataModel &DataModel) {
  Impl->DefaultDataModel = DataModel;
}

mlir::LogicalResult
CliftDialect::verifyOperationAttribute(mlir::Operation *Op,
                                       mlir::NamedAttribute Attr) {
  if (Attr.getName() == getModuleAttrName())
    return verifyModuleAttr(Op, Attr.getValue());

  if (Attr.getName() == getDataModelAttrName())
    return verifyDataModelAttr(Op, Attr.getValue());

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
