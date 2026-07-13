//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "mlir/IR/BuiltinAttributes.h"

#include "revng/Clift/CliftAttributes.h"
#include "revng/Clift/ModuleVisitor.h"
#include "revng/CliftImportModel/Verify.h"
#include "revng/Model/FunctionAttribute.h"
#include "revng/Model/RawFunctionDefinition.h"
#include "revng/Model/TypeDefinition.h"
#include "revng/PTML/CAttributes.h"
#include "revng/Pipeline/Location.h"
#include "revng/Pipes/Ranks.h"

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

      if (auto FT = mlir::dyn_cast<clift::FunctionType>(Type)) {
        if (not llvm::isa<model::CABIFunctionDefinition>(D)
            and not llvm::isa<model::RawFunctionDefinition>(D))
          return error() << "Clift ModuleOp contains a FunctionType with "
                            "an invalid handle: '"
                         << Type.getHandle() << "'";

        ptml::Attributes.assertAnnotationName<"_ABI">();

        bool ABIFound = false;
        for (clift::CAttributeAttr Attr : FT.getCAttributes()) {
          if (not ptml::Attributes.isMacro(Attr.getName().getName()))
            return error() << "Unknown c-attribute ('"
                           << Attr.getName().getName() << "') found in '"
                           << Type.getHandle() << "'";

          if (Attr.getName().getName() == "_ABI") {
            if (std::exchange(ABIFound, true))
              return error() << "Duplicate `_ABI` attributes found in: '"
                             << Type.getHandle() << "'";

            mlir::ArrayAttr Arguments = Attr.getArguments();
            if (not Arguments)
              return error() << "`_ABI` attribute must have an argument. See '"
                             << Type.getHandle() << "'";

            if (Arguments.size() != 1)
              return error() << "`_ABI` attribute must have exactly one "
                                "argument. See '"
                             << Type.getHandle() << "'";

            using clift::CIdentifierAttr;
            auto Identifier = mlir::dyn_cast<CIdentifierAttr>(Arguments[0]);
            if (not Identifier)
              return error() << "`_ABI` attribute argument must be "
                                "an identifier. See '"
                             << Type.getHandle() << "'";

            std::string ModelABI = "raw_";
            if (auto CF = llvm::dyn_cast<model::CABIFunctionDefinition>(&D))
              ModelABI = model::ABI::getName(CF->ABI());
            else if (auto RF = llvm::dyn_cast<model::RawFunctionDefinition>(&D))
              ModelABI += model::Architecture::getName(RF->Architecture());
            else
              revng_abort("Unsupported function type");

            llvm::StringRef AttrABIName = Identifier.getName();
            if (AttrABIName != ModelABI)
              return error() << "`_ABI` attribute value ('" << AttrABIName
                             << "') differs from the model value ('" << ModelABI
                             << "'). See '" << Type.getHandle() << "'";

          } else {
            return error() << "Forbidden c-attribute ('"
                           << Attr.getName().getName() << "') found in '"
                           << Type.getHandle() << "'";
          }
        }

        if (not ABIFound)
          return error() << "`_ABI` attribute is required on every function "
                            "type. See '"
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
      } else if (auto ST = mlir::dyn_cast<clift::StructType>(Type)) {
        if (not llvm::isa<model::StructDefinition>(D))
          return error() << "Clift ModuleOp contains a StructType with "
                            "an invalid handle: '"
                         << Type.getHandle() << "'";

        ptml::Attributes.assertAttributeName<"_CAN_CONTAIN_CODE">();

        bool CodeFound = false;
        for (clift::CAttributeAttr Attr : ST.getCAttributes()) {
          if (not ptml::Attributes.isMacro(Attr.getName().getName()))
            return error() << "Unknown c-attribute ('"
                           << Attr.getName().getName() << "') found in '"
                           << Type.getHandle() << "'";

          if (Attr.getName().getName() == "_CAN_CONTAIN_CODE") {
            if (std::exchange(CodeFound, true))
              return error() << "Duplicate `_CAN_CONTAIN_CODE` attributes "
                                "found in: '"
                             << Type.getHandle() << "'";

            if (Attr.getArguments())
              return error() << "`_CAN_CONTAIN_CODE` attribute must not have "
                                "any arguments. See '"
                             << Type.getHandle() << "'";

          } else {
            return error() << "Forbidden c-attribute ('"
                           << Attr.getName().getName() << "') found in '"
                           << Type.getHandle() << "'";
          }
        }

        bool IsCode = llvm::cast<model::StructDefinition>(D).CanContainCode();
        if (CodeFound != IsCode)
          return error() << "`_CAN_CONTAIN_CODE` status ('" << CodeFound
                         << "') does not match the model value ('" << IsCode
                         << "') for : '" << Type.getHandle() << "'";

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
    } else if (auto L = GetLocation(ranks::OpaqueType)) {
      if (not mlir::isa<clift::StructType>(Type))
        return error() << "Clift ModuleOp contains a non-struct type with "
                          "an OpaqueType handle: '"
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
    const model::TypeDefinition *Prototype = nullptr;
    const model::Function::TypeOfAttributes *Attributes = nullptr;

    if (auto L = GetLocation(ranks::Function)) {
      const auto &[Key] = L->at(ranks::Function);
      auto It = Model.Functions().find(Key);
      if (It == Model.Functions().end())
        return error() << "Clift ModuleOp contains an isolated function with "
                          "an invalid handle: '"
                       << Op.getHandle() << "'";

      Prototype = It->prototype();
      Attributes = &It->Attributes();
      IsIsolated = true;

    } else if (auto L = GetLocation(ranks::DynamicFunction)) {
      const auto &[Key] = L->at(ranks::DynamicFunction);
      auto It = Model.ImportedDynamicFunctions().find(Key);
      if (It == Model.ImportedDynamicFunctions().end())
        return error() << "Clift ModuleOp contains an imported function with "
                          "an invalid handle: '"
                       << Op.getHandle() << "'";

      Prototype = It->prototype();
      Attributes = &It->Attributes();

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

      clift::AttrDictView View = Op.getArgAttrs(Index);
      auto Handle = View.getStringOrEmpty("clift.handle");
      if (Handle.empty())
        Handle = "(a no-handle argument)";

      if (auto CAs = View.getOfType<mlir::ArrayAttr>("clift.c_attributes")) {
        for (mlir::Attribute RawCAttr : CAs) {
          auto CAttribute = mlir::dyn_cast<clift::CAttributeAttr>(RawCAttr);
          if (not CAttribute)
            return error() << "A non c-attribute was found among "
                              "the `c_attributes` in '"
                           << Op.getHandle() << "'";

          auto AttributeName = CAttribute.getName().getName();
          auto Arguments = CAttribute.getArguments();

          if (not ptml::Attributes.isMacro(AttributeName))
            return error() << "Unknown c-attribute ('" << AttributeName
                           << "') found in '" << Handle << "' of '"
                           << Op.getHandle() << "'";

          ptml::Attributes.assertAttributeName<"_STACK">();
          ptml::Attributes.assertAnnotationName<"_REG">();

          if (AttributeName == "_STACK") {
            if (std::exchange(IsStack, true))
              return error() << "More than one `_STACK` attribute is "
                                "attached to '"
                             << Handle << "' of '" << Op.getHandle() << "'";

            if (Arguments)
              return error() << "`_STACK` attribute must not have any "
                                "arguments. See '"
                             << Handle << "' of '" << Op.getHandle() << "'";

            if (Index != Op.getArgCount() - 1)
              return error() << "`_STACK` attribute is only allowed on the "
                                "very last argument. See '"
                             << Handle << "' of '" << Op.getHandle() << "'";
            if (not Prototype)
              return error() << "`_STACK` attribute is only allowed on "
                                "functions with a prototype. See '"
                             << Handle << "' of '" << Op.getHandle() << "'";

            const auto *RFT = Prototype->getRawFunction();
            if (not RFT)
              return error() << "`_STACK` attribute is only allowed on "
                                "functions with a *raw* prototype. See '"
                             << Handle << "' of '" << Op.getHandle() << "'";

            if (not RFT->StackArgumentsType())
              return error() << "`_STACK` attribute is only allowed on "
                                "functions that have stack arguments in the "
                                "model. See '"
                             << Handle << "' of '" << Op.getHandle() << "'";

          } else if (AttributeName == "_REG") {
            if (std::exchange(IsRegister, true))
              return error() << "More than one `_REG` attribute is "
                                "attached to '"
                             << Handle << "' of '" << Op.getHandle() << "'";

            if (not Arguments)
              return error() << "`_REG` attribute must have an argument. See '"
                             << Handle << "' of '" << Op.getHandle() << "'";

            auto ArgumentArray = mlir::cast<mlir::ArrayAttr>(Arguments);
            if (ArgumentArray.size() != 1)
              return error() << "`_REG` attribute must have exactly one "
                                "argument. See '"
                             << Handle << "' of '" << Op.getHandle() << "'";

            using clift::CIdentifierAttr;
            auto Identifier = mlir::dyn_cast<CIdentifierAttr>(Arguments[0]);
            if (not Identifier)
              return error() << "`_REG` attribute argument must be "
                                "an identifier. See '"
                             << Handle << "' of '" << Op.getHandle() << "'";

            if (not Prototype)
              return error() << "`_REG` attribute is only allowed on "
                                "functions with a prototype. See '"
                             << Handle << "' of '" << Op.getHandle() << "'";

            const auto *RFT = Prototype->getRawFunction();
            if (not RFT)
              return error() << "`_REG` attribute is only allowed on "
                                "functions with a *raw* prototype. See '"
                             << Handle << "' of '" << Op.getHandle() << "'";

            if (RFT->Arguments().size() <= Index)
              return error() << "`_REG` attribute is attached to "
                                "a non-existent register argument. See '"
                             << Handle << "' of '" << Op.getHandle() << "'";

            const auto &Argument = *std::next(RFT->Arguments().begin(), Index);
            auto RegisterName = model::Register::getName(Argument.Location());

            llvm::StringRef AttrArgumentName = Identifier.getName();
            if (AttrArgumentName != RegisterName)
              return error() << "`_REG` attribute ('" << AttrArgumentName
                             << "') differs from the model value ('"
                             << RegisterName << "'). See '" << Handle
                             << "' of '" << Op.getHandle() << "'";

          } else {
            return error() << "Forbidden c-attribute ('" << AttributeName
                           << "') found in '" << Handle << "' of '"
                           << Op.getHandle() << "'";
          }

          if (IsStack and IsRegister)
            return error() << "*Both* `_STACK` and `_REG` attributes are "
                              "attached to '"
                           << Handle << "' of '" << Op.getHandle() << "'";
        }
      }
    }

    std::size_t AttributeCount = 0;
    if (Op->hasAttr("noreturn")) {
      if (Attributes == nullptr)
        return error() << "`_NO_RETURN` is attached to a function that does "
                          "not support attributes. See '"
                       << Op.getHandle() << "'";

      if (not Attributes->contains(model::FunctionAttribute::NoReturn))
        return error() << "`_NO_RETURN` is attached to a function that does "
                          "not have it in the model. See '"
                       << Op.getHandle() << "'";

      ++AttributeCount;
    }

    if (Op->hasAttr("always_inline")) {
      if (Attributes == nullptr)
        return error() << "`_ALWAYS_INLINE` is attached to a function that "
                          "does not support attributes. See '"
                       << Op.getHandle() << "'";

      if (not Attributes->contains(model::FunctionAttribute::AlwaysInline))
        return error() << "`_ALWAYS_INLINE` is attached to a function that "
                          "does not have it in the model. See '"
                       << Op.getHandle() << "'";

      ++AttributeCount;
    }

    if (Attributes and AttributeCount != Attributes->size()) {
      return error() << "Attached function attribute count ('" << AttributeCount
                     << "') does not match the model value ('"
                     << Attributes->size() << "'). See '" << Op.getHandle()
                     << "'";
    }

    if (mlir::Attribute RawAttributes = Op->getAttr("clift.c_attributes")) {
      mlir::ArrayAttr Attributes = mlir::cast<mlir::ArrayAttr>(RawAttributes);
      for (mlir::Attribute RawCAttribute : Attributes) {
        auto CAttribute = mlir::dyn_cast<clift::CAttributeAttr>(RawCAttribute);
        if (not CAttribute)
          return error() << "A non c-attribute was found among "
                            "the `c_attributes` in '"
                         << Op.getHandle() << "'";

        if (not ptml::Attributes.isMacro(CAttribute.getName().getName())) {
          return error() << "Unknown c-attribute ('"
                         << CAttribute.getName().getName() << "') found in '"
                         << Op.getHandle() << "'";

        } else {
          return error() << "Forbidden c-attribute ('"
                         << CAttribute.getName().getName() << "') found in '"
                         << Op.getHandle() << "'";
        }
      }
    }

    return mlir::success();
  }

  mlir::LogicalResult visitGlobalVariableOp(clift::GlobalVariableOp Op) {

    if (auto L = pipeline::locationFromString(ranks::Segment, Op.getHandle())) {
      const auto &[Key] = L->at(ranks::Segment);
      auto It = Model.Segments().find(Key);
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
