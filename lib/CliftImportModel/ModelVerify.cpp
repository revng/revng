//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/Clift/ModuleVisitor.h"
#include "revng/CliftImportModel/ModelVerify.h"
#include "revng/Pipeline/Location.h"
#include "revng/Pipes/Ranks.h"

namespace clift = mlir::clift;
namespace ranks = revng::ranks;

namespace {

static constexpr model::PrimitiveKind::Values
integerToPrimitiveKind(clift::IntegerKind Kind) {
  switch (Kind) {
  case clift::IntegerKind::Generic:
    return model::PrimitiveKind::Generic;
  case clift::IntegerKind::PointerOrNumber:
    return model::PrimitiveKind::PointerOrNumber;
  case clift::IntegerKind::Number:
    return model::PrimitiveKind::Number;
  case clift::IntegerKind::Unsigned:
    return model::PrimitiveKind::Unsigned;
  case clift::IntegerKind::Signed:
    return model::PrimitiveKind::Signed;
  default:
    return model::PrimitiveKind::Invalid;
  }
}

static auto getModelPrimitiveType(clift::PrimitiveType Type) {
  if (mlir::isa<clift::VoidType>(Type))
    return model::PrimitiveType::makeVoid();

  if (auto T = mlir::dyn_cast<clift::FloatType>(Type))
    return model::PrimitiveType::make(model::PrimitiveKind::Float, T.getSize());

  auto T = mlir::cast<clift::IntegerType>(Type);
  auto Kind = integerToPrimitiveKind(T.getKind());
  return model::PrimitiveType::make(Kind, T.getSize());
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
  auto error() { return getCurrentOp()->emitError(); }

private:
  mlir::LogicalResult visitDefinedType(clift::DefinedType Type) {
    auto GetLocation = [&](const auto &Rank) {
      return pipeline::locationFromString(Rank, Type.getHandle());
    };

    if (auto L = GetLocation(ranks::TypeDefinition)) {
      auto It = Model.TypeDefinitions().find(L->at(ranks::TypeDefinition));
      if (It == Model.TypeDefinitions().end())
        return error() << "Clift ModuleOp contains a DefinedType with "
                          "an invalid handle: '"
                       << Type.getHandle() << "'";
      const model::TypeDefinition &D = **It;

      if (mlir::isa<clift::FunctionType>(Type)) {
        if (not llvm::isa<model::CABIFunctionDefinition>(D)
            and not llvm::isa<model::RawFunctionDefinition>(D))
          return error() << "Clift ModuleOp contains a FunctionType with "
                            "an invalid handle: '"
                         << Type.getHandle() << "'";
      } else if (mlir::isa<clift::TypedefType>(Type)) {
        if (not llvm::isa<model::TypedefDefinition>(D))
          return error() << "Clift ModuleOp contains a TypedefType with "
                            "an invalid handle: '"
                         << Type.getHandle() << "'";
      } else if (mlir::isa<clift::EnumType>(Type)) {
        if (not llvm::isa<model::EnumDefinition>(D))
          return error() << "Clift ModuleOp contains an EnumType with "
                            "an invalid handle: '"
                         << Type.getHandle() << "'";
      } else if (mlir::isa<clift::StructType>(Type)) {
        if (not llvm::isa<model::StructDefinition>(D))
          return error() << "Clift ModuleOp contains a StructType with "
                            "an invalid handle: '"
                         << Type.getHandle() << "'";
      } else if (mlir::isa<clift::UnionType>(Type)) {
        if (not llvm::isa<model::UnionDefinition>(D))
          return error() << "Clift ModuleOp contains a UnionType with "
                            "an invalid handle: '"
                         << Type.getHandle() << "'";
      }
    } else if (auto L = GetLocation(ranks::HelperStructType)) {
      if (not mlir::isa<clift::StructType>(Type))
        return error() << "Clift ModuleOp contains a non-struct type with "
                          "a HelperStructType handle: '"
                       << Type.getHandle() << "'";
    } else if (auto L = GetLocation(ranks::HelperFunction)) {
      if (not mlir::isa<clift::FunctionType>(Type))
        return error() << "Clift ModuleOp contains a non-function type with "
                          "a HelperFunction handle: '"
                       << Type.getHandle() << "'";
    } else if (auto L = GetLocation(ranks::ArtificialStruct)) {
      if (not mlir::isa<clift::StructType>(Type))
        return error() << "Clift ModuleOp contains a non-struct type with "
                          "an ArtificialStruct handle: '"
                       << Type.getHandle() << "'";
    } else {
      return error() << "Clift ModuleOp contains a DefinedType with "
                        "an invalid handle: '"
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
        return error() << "Clift ModuleOp contains an isolated function with "
                          "an invalid handle: '"
                       << Op.getHandle() << "'";
      IsIsolated = true;
    } else if (auto L = GetLocation(ranks::DynamicFunction)) {
      const auto &[Key] = L->at(ranks::DynamicFunction);
      auto It = Model.ImportedDynamicFunctions().find(Key);
      if (It == Model.ImportedDynamicFunctions().end())
        return error() << "Clift ModuleOp contains an imported function with "
                          "an invalid handle: '"
                       << Op.getHandle() << "'";
    } else if (auto L = GetLocation(ranks::HelperFunction)) {
    } else {
      return error() << "Clift ModuleOp contains a function with an invalid "
                        "handle: "
                        "'"
                     << Op.getHandle() << "'";
    }

    if (not IsIsolated and not Op.isExternal())
      return error() << "Clift ModuleOp contains a non-isolated function with "
                        "a definition: '"
                     << Op.getHandle() << "'";

    for (unsigned Index = 0; Index < Op.getArgCount(); ++Index) {
      bool IsStack = false;
      bool IsRegister = false;

      mlir::clift::AttrDictView View = Op.getArgAttrs(Index);
      auto Handle = View.getStringOrEmpty("clift.handle");
      if (Handle.empty())
        Handle = "(a no-handle argument)";

      if (auto CAs = View.getOfType<mlir::ArrayAttr>("clift.c_attributes")) {
        for (mlir::Attribute CAttribute : CAs) {
          auto AttrName = mlir::cast<clift::CAttributeAttr>(CAttribute)
                            .getName()
                            .getName();

          if (AttrName == "_STACK" and std::exchange(IsStack, true))
            return error() << "More than one _STACK attribute is attached to '"
                           << Handle << "' of '" << Op.getHandle() << "'";

          if (AttrName == "_REG" and std::exchange(IsRegister, true))
            return error() << "More than one _REG attribute is attached to '"
                           << Handle << "' of '" << Op.getHandle() << "'";

          if (IsStack and IsRegister)
            return error() << "*Both* _STACK and _REG attributes are "
                              "attached to '"
                           << Handle << "' of '" << Op.getHandle() << "'";
        }
      }
    }

    return mlir::success();
  }

  mlir::LogicalResult visitGlobalVariableOp(clift::GlobalVariableOp Op) {
    if (auto L = pipeline::locationFromString(ranks::Segment, Op.getHandle())) {
      auto It = Model.Segments().find(L->at(ranks::Segment));
      if (It == Model.Segments().end())
        return error() << "Clift ModuleOp contains a segment with "
                          "an invalid handle: '"
                       << Op.getHandle() << "'";
    } else {
      return error() << "Clift ModuleOp contains a global variable with "
                        "an invalid handle: '"
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
