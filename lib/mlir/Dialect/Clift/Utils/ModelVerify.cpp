//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/Pipeline/Location.h"
#include "revng/Pipes/Ranks.h"
#include "revng/mlir/Dialect/Clift/Utils/ModelVerify.h"
#include "revng/mlir/Dialect/Clift/Utils/ModuleVisitor.h"

namespace clift = mlir::clift;
namespace ranks = revng::ranks;

namespace {

static constexpr model::PrimitiveKind::Values
kindToKind(const clift::PrimitiveKind Kind) {
  return static_cast<model::PrimitiveKind::Values>(Kind);
}

/// Test that kindToKind converts each clift::PrimitiveKind to the matching
/// model::PrimitiveKind. Use a switch converting in the opposite direction
/// in order to produce a warning if a new primitive kind is ever added.
static consteval bool testKindToKind() {
  clift::PrimitiveKind UninitializedKind;
  const auto TestSwitch = [&](const model::PrimitiveKind::Values Kind) {
    switch (Kind) {
    case model::PrimitiveKind::Float:
      return clift::PrimitiveKind::FloatKind;
    case model::PrimitiveKind::Generic:
      return clift::PrimitiveKind::GenericKind;
    case model::PrimitiveKind::Number:
      return clift::PrimitiveKind::NumberKind;
    case model::PrimitiveKind::PointerOrNumber:
      return clift::PrimitiveKind::PointerOrNumberKind;
    case model::PrimitiveKind::Signed:
      return clift::PrimitiveKind::SignedKind;
    case model::PrimitiveKind::Unsigned:
      return clift::PrimitiveKind::UnsignedKind;
    case model::PrimitiveKind::Void:
      return clift::PrimitiveKind::VoidKind;

    case model::PrimitiveKind::Invalid:
    case model::PrimitiveKind::Count:
      // Unreachable. This causes an error during constant evaluation.
      return UninitializedKind;
    }
  };

  for (int I = 0; I < static_cast<int>(model::PrimitiveKind::Count); ++I) {
    auto const Kind = static_cast<model::PrimitiveKind::Values>(I);
    if (Kind != model::PrimitiveKind::Invalid) {
      if (kindToKind(TestSwitch(Kind)) != Kind)
        return false;
    }
  }
  return true;
}
static_assert(testKindToKind());

static auto getModelPrimitiveType(clift::PrimitiveType T) {
  return model::PrimitiveType::make(kindToKind(T.getKind()), T.getSize());
}

class Verifier : public clift::ModuleVisitor<Verifier> {
public:
  explicit Verifier(const model::Binary &Model) : Model(Model) {}

  mlir::LogicalResult visitType(mlir::Type Type) {
    if (auto T = mlir::dyn_cast<clift::PrimitiveType>(Type)) {
      if (not getModelPrimitiveType(T)->verify())
        return mlir::failure();
    } else if (auto T = mlir::dyn_cast<clift::DefinedType>(Type)) {
      if (visitDefinedType(T).failed())
        return mlir::failure();
    }
    return mlir::success();
  }

  mlir::LogicalResult visitNestedOp(mlir::Operation *Op) {
    if (auto F = mlir::dyn_cast<clift::FunctionOp>(Op)) {
      if (visitFunctionOp(F).failed())
        return mlir::failure();
    } else if (auto G = mlir::dyn_cast<clift::GlobalVariableOp>(Op)) {
      if (visitGlobalVariableOp(G).failed())
        return mlir::failure();
    }
    return mlir::success();
  }

  mlir::LogicalResult visitModuleLevelOp(mlir::Operation *Op) {
    if (auto F = mlir::dyn_cast<clift::FunctionOp>(Op))
      return visitFunctionOp(F);
    if (auto G = mlir::dyn_cast<clift::GlobalVariableOp>(Op))
      return visitGlobalVariableOp(G);
    return mlir::success();
  }

private:
  mlir::LogicalResult visitDefinedType(clift::DefinedType Type) {
    auto GetLocation = [&](const auto &Rank) {
      return pipeline::locationFromString(Rank, Type.getHandle());
    };

    if (auto L = GetLocation(ranks::TypeDefinition)) {
      auto It = Model.TypeDefinitions().find(L->at(ranks::TypeDefinition));
      if (It == Model.TypeDefinitions().end())
        return getCurrentOp()->emitError() << "Clift ModuleOp contains "
                                              "DefinedType with invalid "
                                              "handle: '"
                                           << Type.getHandle() << "'";
      const model::TypeDefinition &D = **It;

      if (mlir::isa<clift::FunctionType>(Type)) {
        if (not llvm::isa<model::CABIFunctionDefinition>(D)
            and not llvm::isa<model::RawFunctionDefinition>(D))
          return getCurrentOp()->emitError() << "Clift ModuleOp contains "
                                                "FunctionType with invalid "
                                                "handle: '"
                                             << Type.getHandle() << "'";
      } else if (mlir::isa<clift::TypedefType>(Type)) {
        if (not llvm::isa<model::TypedefDefinition>(D))
          return getCurrentOp()->emitError() << "Clift ModuleOp contains "
                                                "TypedefType with invalid "
                                                "handle: '"
                                             << Type.getHandle() << "'";
      } else if (mlir::isa<clift::EnumType>(Type)) {
        if (not llvm::isa<model::EnumDefinition>(D))
          return getCurrentOp()->emitError() << "Clift ModuleOp contains "
                                                "EnumType with invalid "
                                                "handle: '"
                                             << Type.getHandle() << "'";
      } else if (mlir::isa<clift::StructType>(Type)) {
        if (not llvm::isa<model::StructDefinition>(D))
          return getCurrentOp()->emitError() << "Clift ModuleOp contains "
                                                "StructType with invalid "
                                                "handle: '"
                                             << Type.getHandle() << "'";
      } else if (mlir::isa<clift::UnionType>(Type)) {
        if (not llvm::isa<model::UnionDefinition>(D))
          return getCurrentOp()->emitError() << "Clift ModuleOp contains "
                                                "UnionType with invalid "
                                                "handle: '"
                                             << Type.getHandle() << "'";
      }
    } else if (auto L = GetLocation(ranks::HelperStructType)) {
      if (not mlir::isa<clift::StructType>(Type))
        return getCurrentOp()->emitError() << "Clift ModuleOp contains "
                                              "non-struct type with "
                                              "HelperStructType handle: '"
                                           << Type.getHandle() << "'";
    } else if (auto L = GetLocation(ranks::HelperFunction)) {
      if (not mlir::isa<clift::FunctionType>(Type))
        return getCurrentOp()->emitError() << "Clift ModuleOp contains "
                                              "non-function type with "
                                              "HelperFunction handle: '"
                                           << Type.getHandle() << "'";
    } else if (auto L = GetLocation(ranks::ArtificialStruct)) {
      if (not mlir::isa<clift::StructType>(Type))
        return getCurrentOp()->emitError() << "Clift ModuleOp contains "
                                              "non-struct type with "
                                              "ArtificialStruct handle: '"
                                           << Type.getHandle() << "'";
    } else {
      return getCurrentOp()->emitError() << "Clift ModuleOp contains "
                                            "DefinedType with invalid handle: '"
                                         << Type.getHandle() << "'";
    }

    return mlir::success();
  }

  mlir::LogicalResult visitFunctionOp(clift::FunctionOp Op) {
    auto GetLocation = [&](const auto &Rank) {
      return pipeline::locationFromString(Rank, Op.getHandle());
    };

    bool IsIsolated = false;
    if (auto L = GetLocation(ranks::Function)) {
      const auto &[Key] = L->at(ranks::Function);
      auto It = Model.Functions().find(Key);
      if (It == Model.Functions().end())
        return getCurrentOp()->emitError() << "Clift ModuleOp contains "
                                              "function with invalid isolated "
                                              "handle: '"
                                           << Op.getHandle() << "'";
      IsIsolated = true;
    } else if (auto L = GetLocation(ranks::DynamicFunction)) {
      const auto &[Key] = L->at(ranks::DynamicFunction);
      auto It = Model.ImportedDynamicFunctions().find(Key);
      if (It == Model.ImportedDynamicFunctions().end())
        return getCurrentOp()->emitError() << "Clift ModuleOp contains "
                                              "function with invalid imported "
                                              "handle: '"
                                           << Op.getHandle() << "'";
    } else if (auto L = GetLocation(ranks::HelperFunction)) {
    } else {
      return getCurrentOp()->emitError() << "Clift ModuleOp contains function "
                                            "with invalid handle: '"
                                         << Op.getHandle() << "'";
    }

    if (not IsIsolated and not Op.isExternal())
      return getCurrentOp()->emitError() << "Clift ModuleOp contains "
                                            "non-isolated function with a "
                                            "definition: '"
                                         << Op.getHandle() << "'";

    return mlir::success();
  }

  mlir::LogicalResult visitGlobalVariableOp(clift::GlobalVariableOp Op) {
    if (auto L = pipeline::locationFromString(ranks::Segment, Op.getHandle())) {
      auto It = Model.Segments().find(L->at(ranks::Segment));
      if (It == Model.Segments().end())
        return getCurrentOp()->emitError() << "Clift ModuleOp contains global "
                                              "variable with invalid segment "
                                              "handle: '"
                                           << Op.getHandle() << "'";
    } else {
      return getCurrentOp()->emitError() << "Clift ModuleOp contains global "
                                            "variable with invalid handle: '"
                                         << Op.getHandle() << "'";
    }

    return mlir::success();
  }

  const model::Binary &Model;
};

} // namespace

mlir::LogicalResult clift::verifyAgainstModel(mlir::ModuleOp Module,
                                              const model::Binary &Model) {
  return Verifier::visit(Module, Model);
}
