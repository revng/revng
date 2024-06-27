//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/ADT/SmallSet.h"
#include "llvm/Support/FormatVariadic.h"

#include "revng/ADT/RecursiveCoroutine.h"

#include "revng-c/mlir/Dialect/Clift/IR/Clift.h"
#include "revng-c/mlir/Dialect/Clift/IR/CliftAttributes.h"
#include "revng-c/mlir/Dialect/Clift/IR/CliftTypes.h"
#include "revng-c/mlir/Dialect/Clift/Utils/ImportModel.h"

namespace {

namespace clift = mlir::clift;

template<typename Attribute>
using AttributeVector = llvm::SmallVector<Attribute, 16>;

class CliftConverter {
  mlir::MLIRContext *Context;
  llvm::function_ref<mlir::InFlightDiagnostic()> EmitError;

  llvm::DenseMap<uint64_t, clift::TypeDefinitionAttr> Cache;
  llvm::DenseMap<uint64_t, const model::TypeDefinition *> IncompleteTypes;

  llvm::SmallSet<uint64_t, 16> DefinitionGuardSet;

  class RecursiveDefinitionGuard {
    CliftConverter *Self = nullptr;
    uint64_t ID;

  public:
    explicit RecursiveDefinitionGuard(CliftConverter &Self, const uint64_t ID) {
      if (Self.DefinitionGuardSet.insert(ID).second) {
        this->Self = &Self;
        this->ID = ID;
      }
    }

    RecursiveDefinitionGuard(const RecursiveDefinitionGuard &) = delete;
    RecursiveDefinitionGuard &
    operator=(const RecursiveDefinitionGuard &) = delete;

    ~RecursiveDefinitionGuard() {
      if (Self != nullptr) {
        size_t const Erased = Self->DefinitionGuardSet.erase(ID);
        revng_assert(Erased == 1);
      }
    }

    explicit operator bool() const { return Self != nullptr; }
  };

public:
  explicit CliftConverter(mlir::MLIRContext &Context,
                          llvm::function_ref<mlir::InFlightDiagnostic()>
                            EmitError) :
    Context(&Context), EmitError(EmitError) {}

  CliftConverter(const CliftConverter &) = delete;
  CliftConverter &operator=(const CliftConverter &) = delete;

  ~CliftConverter() { revng_assert(DefinitionGuardSet.empty()); }

  clift::ValueType
  convertTypeDefinition(const model::TypeDefinition &ModelType) {
    const clift::ValueType T = fromTypeDefinition(ModelType,
                                                  /* RequireComplete = */ true);
    if (T and not processIncompleteTypes())
      return nullptr;
    return T;
  }

  clift::ValueType convertType(const model::Type &ModelType) {
    const clift::ValueType T = fromType(ModelType,
                                        /* RequireComplete = */ true);
    if (T and not processIncompleteTypes())
      return nullptr;
    return T;
  }

private:
  mlir::BoolAttr getBool(bool const Value) {
    return mlir::BoolAttr::get(Context, Value);
  }

  mlir::BoolAttr getFalse() { return getBool(false); }

  template<typename T, typename... ArgTypes>
  T make(const ArgTypes &...Args) {
    if (failed(T::verify(EmitError, Args...)))
      return {};
    return T::get(Context, Args...);
  }

  static clift::PrimitiveKind
  getPrimitiveKind(const model::PrimitiveType &ModelType) {
    switch (ModelType.PrimitiveKind()) {
    case model::PrimitiveKind::Void:
      return clift::PrimitiveKind::VoidKind;
    case model::PrimitiveKind::Generic:
      return clift::PrimitiveKind::GenericKind;
    case model::PrimitiveKind::PointerOrNumber:
      return clift::PrimitiveKind::PointerOrNumberKind;
    case model::PrimitiveKind::Number:
      return clift::PrimitiveKind::NumberKind;
    case model::PrimitiveKind::Unsigned:
      return clift::PrimitiveKind::UnsignedKind;
    case model::PrimitiveKind::Signed:
      return clift::PrimitiveKind::SignedKind;
    case model::PrimitiveKind::Float:
      return clift::PrimitiveKind::FloatKind;

    case model::PrimitiveKind::Invalid:
    case model::PrimitiveKind::Count:
      revng_abort("These are invalid values. Something has gone wrong.");
    }
  }

  RecursiveCoroutine<clift::TypeDefinitionAttr>
  getTypeAttribute(const model::CABIFunctionDefinition &ModelType) {
    RecursiveDefinitionGuard Guard(*this, ModelType.ID());
    if (not Guard) {
      if (EmitError)
        EmitError() << "Recursive definition of CABIFunctionDefinition "
                    << ModelType.ID();
      rc_return nullptr;
    }

    AttributeVector<clift::FunctionArgumentAttr> Args;
    Args.reserve(ModelType.Arguments().size());

    for (const model::Argument &Argument : ModelType.Arguments()) {
      const auto Type = rc_recur fromType(*Argument.Type());
      if (not Type)
        rc_return nullptr;
      const llvm::StringRef Name = Argument.name();
      const auto Attribute = make<clift::FunctionArgumentAttr>(Type, Name);
      if (not Attribute)
        rc_return nullptr;
      Args.push_back(Attribute);
    }

    clift::ValueType ReturnType = nullptr;
    if (ModelType.ReturnType().isEmpty())
      ReturnType = rc_recur fromType(*model::PrimitiveType::makeVoid());
    else
      ReturnType = rc_recur fromType(*ModelType.ReturnType());
    if (not ReturnType)
      rc_return nullptr;

    rc_return make<clift::FunctionTypeAttr>(ModelType.ID(),
                                            ModelType.name(),
                                            ReturnType,
                                            Args);
  }

  RecursiveCoroutine<clift::TypeDefinitionAttr>
  getTypeAttribute(const model::EnumDefinition &ModelType) {
    RecursiveDefinitionGuard Guard(*this, ModelType.ID());
    if (not Guard) {
      if (EmitError)
        EmitError() << "Recursive definition of EnumDefinition "
                    << ModelType.ID();
      rc_return nullptr;
    }

    const auto UnderlyingType = rc_recur fromType(*ModelType.UnderlyingType());
    if (not UnderlyingType)
      rc_return nullptr;

    AttributeVector<clift::EnumFieldAttr> Fields;
    Fields.reserve(ModelType.Entries().size());

    for (const model::EnumEntry &Entry : ModelType.Entries()) {
      const auto Attribute = make<clift::EnumFieldAttr>(Entry.Value(),
                                                        ModelType
                                                          .entryName(Entry));
      if (not Attribute)
        rc_return nullptr;
      Fields.push_back(Attribute);
    }

    rc_return make<clift::EnumTypeAttr>(ModelType.ID(),
                                        ModelType.name(),
                                        UnderlyingType,
                                        Fields);
  }

  RecursiveCoroutine<clift::ValueType>
  getScalarTupleType(const model::RawFunctionDefinition &ModelType) {
    using ElementAttr = clift::ScalarTupleElementAttr;

    AttributeVector<ElementAttr> Elements;
    Elements.reserve(ModelType.ReturnValues().size());

    for (const model::NamedTypedRegister &Register : ModelType.ReturnValues()) {
      const auto RegisterType = rc_recur fromType(*Register.Type());
      if (not RegisterType)
        rc_return nullptr;

      const auto Attribute = make<ElementAttr>(RegisterType, Register.name());
      if (not Attribute)
        rc_return nullptr;

      Elements.push_back(Attribute);
    }

    rc_return make<clift::ScalarTupleType>(ModelType.ID(), "", Elements);
  }

  RecursiveCoroutine<clift::TypeDefinitionAttr>
  getTypeAttribute(const model::RawFunctionDefinition &ModelType) {
    RecursiveDefinitionGuard Guard(*this, ModelType.ID());
    if (not Guard) {
      if (EmitError)
        EmitError() << "Recursive definition of RawFunctionDefinition "
                    << ModelType.ID();
      rc_return nullptr;
    }

    clift::FunctionArgumentAttr StackArgument;
    size_t ArgumentsCount = 0;

    if (not ModelType.StackArgumentsType().isEmpty()) {
      const auto Type = rc_recur fromType(*ModelType.StackArgumentsType());
      if (not Type)
        rc_return nullptr;

      const uint64_t PointerSize = getPointerSize(ModelType.Architecture());
      const auto PointerType = make<clift::PointerType>(Type,
                                                        PointerSize,
                                                        getFalse());
      if (not PointerType)
        rc_return nullptr;

      StackArgument = make<clift::FunctionArgumentAttr>(PointerType, "");
      if (not StackArgument)
        rc_return nullptr;

      ++ArgumentsCount;
    }

    ArgumentsCount += ModelType.Arguments().size();
    AttributeVector<clift::FunctionArgumentAttr> Args;
    Args.reserve(ArgumentsCount);

    for (const model::NamedTypedRegister &Register : ModelType.Arguments()) {
      const auto Type = rc_recur fromType(*Register.Type());
      if (not Type)
        rc_return nullptr;
      const llvm::StringRef Name = Register.name();
      const auto Argument = make<clift::FunctionArgumentAttr>(Type, Name);
      if (not Argument)
        rc_return nullptr;
      Args.push_back(Argument);
    }

    if (StackArgument)
      Args.push_back(StackArgument);

    clift::ValueType ReturnType;
    switch (ModelType.ReturnValues().size()) {
    case 0:
      ReturnType = make<clift::PrimitiveType>(clift::PrimitiveKind::VoidKind,
                                              0,
                                              getFalse());
      break;

    case 1:
      ReturnType = rc_recur fromType(*ModelType.ReturnValues().begin()->Type());
      break;

    default:
      ReturnType = make<clift::ScalarTupleType>(ModelType.ID());
      {
        const auto R = IncompleteTypes.try_emplace(ModelType.ID(), &ModelType);
        revng_assert(R.second && "Scalar tuple types are only visited once.");
      }
      break;
    }
    if (not ReturnType)
      rc_return nullptr;

    rc_return make<clift::FunctionTypeAttr>(ModelType.ID(),
                                            ModelType.name(),
                                            ReturnType,
                                            Args);
  }

  RecursiveCoroutine<clift::TypeDefinitionAttr>
  getTypeAttribute(const model::StructDefinition &ModelType,
                   const bool RequireComplete) {
    if (not RequireComplete) {
      const auto T = clift::StructTypeAttr::get(Context, ModelType.ID());
      if (not T.isDefinition())
        IncompleteTypes.try_emplace(ModelType.ID(), &ModelType);
      rc_return T;
    }

    RecursiveDefinitionGuard Guard(*this, ModelType.ID());
    if (not Guard) {
      if (EmitError)
        EmitError() << "Recursive definition of StructTypeAttr "
                    << ModelType.ID();
      rc_return nullptr;
    }

    AttributeVector<clift::FieldAttr> Fields;
    Fields.reserve(ModelType.Fields().size());

    for (const model::StructField &Field : ModelType.Fields()) {
      const auto FieldType = rc_recur fromType(*Field.Type(),
                                               /* RequireComplete = */ true);
      if (not FieldType)
        rc_return nullptr;
      const auto Attribute = make<clift::FieldAttr>(Field.Offset(),
                                                    FieldType,
                                                    Field.name());
      if (not Attribute)
        rc_return nullptr;
      Fields.push_back(Attribute);
    }

    rc_return make<clift::StructTypeAttr>(ModelType.ID(),
                                          ModelType.name(),
                                          ModelType.Size(),
                                          Fields);
  }

  RecursiveCoroutine<clift::TypeDefinitionAttr>
  getTypeAttribute(const model::TypedefDefinition &ModelType,
                   const bool RequireComplete) {
    std::optional<RecursiveDefinitionGuard> Guard;

    if (RequireComplete) {
      Guard.emplace(*this, ModelType.ID());
      if (not *Guard) {
        if (EmitError)
          EmitError() << "Recursive definition of TypedefDefinition "
                      << ModelType.ID();
        rc_return nullptr;
      }
    }

    const auto UnderlyingType = rc_recur fromType(*ModelType.UnderlyingType(),
                                                  RequireComplete);
    if (not UnderlyingType)
      rc_return nullptr;
    rc_return make<clift::TypedefTypeAttr>(ModelType.ID(),
                                           ModelType.name(),
                                           UnderlyingType);
  }

  RecursiveCoroutine<clift::TypeDefinitionAttr>
  getTypeAttribute(const model::UnionDefinition &ModelType,
                   const bool RequireComplete) {
    if (not RequireComplete) {
      const auto T = clift::UnionTypeAttr::get(Context, ModelType.ID());
      if (not T.isDefinition())
        IncompleteTypes.try_emplace(ModelType.ID(), &ModelType);
      rc_return T;
    }

    RecursiveDefinitionGuard Guard(*this, ModelType.ID());
    if (not Guard) {
      if (EmitError)
        EmitError() << "Recursive definition of UnionTypeAttr "
                    << ModelType.ID();
      rc_return nullptr;
    }

    AttributeVector<clift::FieldAttr> Fields;
    Fields.reserve(ModelType.Fields().size());

    for (const model::UnionField &Field : ModelType.Fields()) {
      const auto FieldType = rc_recur fromType(*Field.Type(),
                                               /* RequireComplete = */ true);
      if (not FieldType)
        rc_return nullptr;
      const auto Attribute = make<clift::FieldAttr>(0, FieldType, Field.name());
      if (not Attribute)
        rc_return nullptr;
      Fields.push_back(Attribute);
    }

    rc_return make<clift::UnionTypeAttr>(ModelType.ID(),
                                         ModelType.name(),
                                         Fields);
  }

  RecursiveCoroutine<clift::TypeDefinitionAttr>
  getTypeAttribute(const model::TypeDefinition &T, bool &RequireComplete) {
    if (const auto *CFT = llvm::dyn_cast<model::CABIFunctionDefinition>(&T))
      rc_return getTypeAttribute(*CFT);

    if (const auto *RFT = llvm::dyn_cast<model::RawFunctionDefinition>(&T))
      rc_return getTypeAttribute(*RFT);

    if (const auto *Enum = llvm::dyn_cast<model::EnumDefinition>(&T))
      rc_return getTypeAttribute(*Enum);

    if (const auto *Struct = llvm::dyn_cast<model::StructDefinition>(&T))
      rc_return getTypeAttribute(*Struct, RequireComplete);

    if (const auto *Union = llvm::dyn_cast<model::UnionDefinition>(&T))
      rc_return getTypeAttribute(*Union, RequireComplete);

    if (const auto *Typedef = llvm::dyn_cast<model::TypedefDefinition>(&T))
      rc_return getTypeAttribute(*Typedef, RequireComplete);

    revng_abort("Unsupported type definition kind.");
  }

  RecursiveCoroutine<clift::ValueType>
  fromTypeDefinition(const model::TypeDefinition &ModelType,
                     bool RequireComplete = false,
                     const bool Const = false) {
    if (const auto It = Cache.find(ModelType.ID()); It != Cache.end())
      rc_return make<clift::DefinedType>(It->second, getBool(Const));

    if (not ModelType.verify()) {
      if (EmitError)
        EmitError() << "Invalid model type definition";

      rc_return nullptr;
    }

    const clift::TypeDefinitionAttr Attr = getTypeAttribute(ModelType,
                                                            RequireComplete);

    if (not Attr)
      rc_return nullptr;

    if (RequireComplete) {
      const auto R = Cache.try_emplace(ModelType.ID(), Attr);
      revng_assert(R.second);
    }

    rc_return make<clift::DefinedType>(Attr, getBool(Const));
  }

  RecursiveCoroutine<clift::ValueType> fromType(const model::Type &ModelType,
                                                bool RequireComplete = false) {
    if (not ModelType.verify()) {
      if (EmitError)
        EmitError() << "Invalid model type";

      rc_return nullptr;
    }

    if (const auto &P = llvm::dyn_cast<model::PrimitiveType>(&ModelType)) {
      rc_return make<clift::PrimitiveType>(getPrimitiveKind(*P),
                                           P->Size(),
                                           getBool(P->IsConst()));

    } else if (const auto &D = llvm::dyn_cast<model::DefinedType>(&ModelType)) {
      rc_return fromTypeDefinition(D->unwrap(), RequireComplete, D->IsConst());

    } else if (const auto &A = llvm::dyn_cast<model::ArrayType>(&ModelType)) {
      rc_return make<clift::ArrayType>(rc_recur fromType(*A->ElementType(),
                                                         RequireComplete),
                                       A->ElementCount(),
                                       getFalse());

    } else if (const auto &P = llvm::dyn_cast<model::PointerType>(&ModelType)) {
      // If there's a pointer in the way, the base type does not have to be
      // complete.
      RequireComplete = false;

      rc_return make<clift::PointerType>(rc_recur fromType(*P->PointeeType(),
                                                           RequireComplete),
                                         P->PointerSize(),
                                         getBool(P->IsConst()));

    } else {
      if (EmitError)
        EmitError() << "Unknown model type";

      rc_return nullptr;
    }
  }

  bool processIncompleteTypes() {
    while (not IncompleteTypes.empty()) {
      const auto Iterator = IncompleteTypes.begin();
      const model::TypeDefinition &ModelType = *Iterator->second;
      IncompleteTypes.erase(Iterator);

      clift::ValueType CompleteType;
      if (const auto
            RFT = llvm::dyn_cast<model::RawFunctionDefinition>(&ModelType)) {
        CompleteType = getScalarTupleType(*RFT);
      } else {
        CompleteType = fromTypeDefinition(ModelType,
                                          /* RequireComplete = */ true);
      }

      if (not CompleteType)
        return false;
    }

    return true;
  }
};

} // namespace

clift::ValueType
clift::importModelType(llvm::function_ref<mlir::InFlightDiagnostic()> EmitError,
                       mlir::MLIRContext &Context,
                       const model::TypeDefinition &ModelType) {
  return CliftConverter(Context, EmitError).convertTypeDefinition(ModelType);
}

clift::ValueType
clift::importModelType(llvm::function_ref<mlir::InFlightDiagnostic()> EmitError,
                       mlir::MLIRContext &Context,
                       const model::Type &ModelType) {
  return CliftConverter(Context, EmitError).convertType(ModelType);
}
