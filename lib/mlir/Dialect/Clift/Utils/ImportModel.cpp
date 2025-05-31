//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/ADT/SmallSet.h"
#include "llvm/Support/FormatVariadic.h"

#include "revng/ADT/RecursiveCoroutine.h"
#include "revng/Model/NameBuilder.h"
#include "revng/Pipeline/Location.h"
#include "revng/Pipes/Ranks.h"
#include "revng/mlir/Dialect/Clift/IR/Clift.h"
#include "revng/mlir/Dialect/Clift/IR/CliftAttributes.h"
#include "revng/mlir/Dialect/Clift/IR/CliftTypes.h"
#include "revng/mlir/Dialect/Clift/Utils/ImportModel.h"

namespace {

namespace clift = mlir::clift;

template<typename Attribute>
using AttributeVector = llvm::SmallVector<Attribute, 16>;

class CliftConverter {
  mlir::MLIRContext *Context;
  model::CNameBuilder NameBuilder;
  llvm::function_ref<mlir::InFlightDiagnostic()> EmitError;

  llvm::DenseMap<uint64_t, clift::DefinedType> Cache;
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
                          const model::Binary &Binary,
                          llvm::function_ref<mlir::InFlightDiagnostic()>
                            EmitError) :
    Context(&Context), NameBuilder(Binary), EmitError(EmitError) {}

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

  std::string getHandle(const model::TypeDefinition &T) {
    return pipeline::locationString(revng::ranks::TypeDefinition, T.key());
  }

  std::string getRegisterSetLocation(const model::RawFunctionDefinition &T) {
    return pipeline::locationString(revng::ranks::ArtificialStruct, T.key());
  }

  RecursiveCoroutine<clift::DefinedType>
  getTypeDefinition(const model::CABIFunctionDefinition &ModelType) {
    RecursiveDefinitionGuard Guard(*this, ModelType.ID());
    if (not Guard) {
      if (EmitError)
        EmitError() << "Recursive definition of CABIFunctionDefinition "
                    << ModelType.ID();
      rc_return nullptr;
    }

    AttributeVector<mlir::Type> ArgumentTypes;
    ArgumentTypes.reserve(ModelType.Arguments().size());

    for (const model::Argument &Argument : ModelType.Arguments()) {
      const auto Type = rc_recur fromType(*Argument.Type());
      if (not Type)
        rc_return nullptr;
      ArgumentTypes.push_back(Type);
    }

    mlir::Type ReturnType = nullptr;
    if (ModelType.ReturnType().isEmpty())
      ReturnType = rc_recur fromType(*model::PrimitiveType::makeVoid());
    else
      ReturnType = rc_recur fromType(*ModelType.ReturnType());
    if (not ReturnType)
      rc_return nullptr;

    rc_return make<clift::FunctionType>(getHandle(ModelType),
                                        NameBuilder.name(ModelType),
                                        ReturnType,
                                        ArgumentTypes);
  }

  RecursiveCoroutine<clift::DefinedType>
  getTypeDefinition(const model::EnumDefinition &ModelType) {
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
      std::string Name = NameBuilder.name(ModelType, Entry);
      const auto Attribute = make<clift::EnumFieldAttr>(Entry.Value(), Name);
      if (not Attribute)
        rc_return nullptr;
      Fields.push_back(Attribute);
    }

    rc_return make<clift::EnumType>(getHandle(ModelType),
                                    NameBuilder.name(ModelType),
                                    UnderlyingType,
                                    Fields);
  }

  RecursiveCoroutine<clift::ValueType>
  getRegisterSetType(const model::RawFunctionDefinition &ModelType) {
    using ElementAttr = clift::FieldAttr;

    AttributeVector<ElementAttr> Elements;
    Elements.reserve(ModelType.ReturnValues().size());

    uint64_t Offset = 0;
    for (const model::NamedTypedRegister &Register : ModelType.ReturnValues()) {
      const auto RegisterType = rc_recur fromType(*Register.Type());
      if (not RegisterType)
        rc_return nullptr;

      const auto Attribute = make<ElementAttr>(Offset,
                                               RegisterType,
                                               NameBuilder.name(ModelType,
                                                                Register));
      if (not Attribute)
        rc_return nullptr;

      Elements.push_back(Attribute);
      Offset += RegisterType.getByteSize();
    }

    std::string TypeName;
    {
      llvm::raw_string_ostream Out(TypeName);
      Out << "register_set_" << ModelType.ID();
    }

    rc_return make<clift::StructType>(getRegisterSetLocation(ModelType),
                                      TypeName,
                                      Offset,
                                      Elements);
  }

  RecursiveCoroutine<clift::DefinedType>
  getTypeDefinition(const model::RawFunctionDefinition &ModelType) {
    RecursiveDefinitionGuard Guard(*this, ModelType.ID());
    if (not Guard) {
      if (EmitError)
        EmitError() << "Recursive definition of RawFunctionDefinition "
                    << ModelType.ID();
      rc_return nullptr;
    }

    mlir::Type StackArgumentType;
    size_t ArgumentsCount = 0;

    if (not ModelType.StackArgumentsType().isEmpty()) {
      const auto Type = rc_recur fromType(*ModelType.StackArgumentsType());
      if (not Type)
        rc_return nullptr;

      const uint64_t PointerSize = getPointerSize(ModelType.Architecture());
      StackArgumentType = make<clift::PointerType>(Type,
                                                   PointerSize,
                                                   /*IsConst=*/false);
      if (not StackArgumentType)
        rc_return nullptr;

      ++ArgumentsCount;
    }

    ArgumentsCount += ModelType.Arguments().size();
    AttributeVector<mlir::Type> ArgumentTypes;
    ArgumentTypes.reserve(ArgumentsCount);

    for (const model::NamedTypedRegister &Register : ModelType.Arguments()) {
      const auto Type = rc_recur fromType(*Register.Type());
      if (not Type)
        rc_return nullptr;
      ArgumentTypes.push_back(Type);
    }

    if (StackArgumentType)
      ArgumentTypes.push_back(StackArgumentType);

    clift::ValueType ReturnType;
    switch (ModelType.ReturnValues().size()) {
    case 0:
      ReturnType = make<clift::PrimitiveType>(clift::PrimitiveKind::VoidKind,
                                              0,
                                              /*IsConst=*/false);
      break;

    case 1:
      ReturnType = rc_recur fromType(*ModelType.ReturnValues().begin()->Type());
      break;

    default: {
      ReturnType = make<clift::StructType>(getRegisterSetLocation(ModelType));
      const auto R = IncompleteTypes.try_emplace(ModelType.ID(), &ModelType);
      revng_assert(R.second && "Register set types are only visited once.");
    } break;
    }
    if (not ReturnType)
      rc_return nullptr;

    rc_return make<clift::FunctionType>(getHandle(ModelType),
                                        NameBuilder.name(ModelType),
                                        ReturnType,
                                        ArgumentTypes);
  }

  RecursiveCoroutine<clift::DefinedType>
  getTypeDefinition(const model::StructDefinition &ModelType,
                    const bool RequireComplete) {
    if (not RequireComplete) {
      const auto T = clift::StructType::get(Context, getHandle(ModelType));
      if (not T.isComplete())
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
      auto Attribute = make<clift::FieldAttr>(Field.Offset(),
                                              FieldType,
                                              NameBuilder.name(ModelType,
                                                               Field));
      if (not Attribute)
        rc_return nullptr;
      Fields.push_back(Attribute);
    }

    rc_return make<clift::StructType>(getHandle(ModelType),
                                      NameBuilder.name(ModelType),
                                      ModelType.Size(),
                                      Fields);
  }

  RecursiveCoroutine<clift::DefinedType>
  getTypeDefinition(const model::TypedefDefinition &ModelType,
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
    rc_return make<clift::TypedefType>(getHandle(ModelType),
                                       NameBuilder.name(ModelType),
                                       UnderlyingType);
  }

  RecursiveCoroutine<clift::DefinedType>
  getTypeDefinition(const model::UnionDefinition &ModelType,
                    const bool RequireComplete) {
    if (not RequireComplete) {
      const auto T = clift::UnionType::get(Context, getHandle(ModelType));
      if (not T.isComplete())
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
      auto Attribute = make<clift::FieldAttr>(0,
                                              FieldType,
                                              NameBuilder.name(ModelType,
                                                               Field));
      if (not Attribute)
        rc_return nullptr;
      Fields.push_back(Attribute);
    }

    rc_return make<clift::UnionType>(getHandle(ModelType),
                                     NameBuilder.name(ModelType),
                                     Fields);
  }

  RecursiveCoroutine<clift::DefinedType>
  getTypeDefinition(const model::TypeDefinition &T, bool &RequireComplete) {
    if (const auto *CFT = llvm::dyn_cast<model::CABIFunctionDefinition>(&T))
      rc_return getTypeDefinition(*CFT);

    if (const auto *RFT = llvm::dyn_cast<model::RawFunctionDefinition>(&T))
      rc_return getTypeDefinition(*RFT);

    if (const auto *Enum = llvm::dyn_cast<model::EnumDefinition>(&T))
      rc_return getTypeDefinition(*Enum);

    if (const auto *Struct = llvm::dyn_cast<model::StructDefinition>(&T))
      rc_return getTypeDefinition(*Struct, RequireComplete);

    if (const auto *Union = llvm::dyn_cast<model::UnionDefinition>(&T))
      rc_return getTypeDefinition(*Union, RequireComplete);

    if (const auto *Typedef = llvm::dyn_cast<model::TypedefDefinition>(&T))
      rc_return getTypeDefinition(*Typedef, RequireComplete);

    revng_abort("Unsupported type definition kind.");
  }

  RecursiveCoroutine<clift::ValueType>
  fromTypeDefinition(const model::TypeDefinition &ModelType,
                     bool RequireComplete = false,
                     const bool Const = false) {
    if (Const) {
      auto Type = rc_recur fromTypeDefinition(ModelType,
                                              RequireComplete,
                                              /*Const=*/false);
      rc_return Type.addConst();
    }

    if (const auto It = Cache.find(ModelType.ID()); It != Cache.end())
      rc_return It->second;

    if (not ModelType.verify()) {
      if (EmitError)
        EmitError() << "Invalid model type definition";

      rc_return nullptr;
    }

    auto Type = rc_recur getTypeDefinition(ModelType, RequireComplete);

    if (Type and RequireComplete) {
      auto [Iterator, Inserted] = Cache.try_emplace(ModelType.ID(), Type);
      revng_assert(Inserted);
    }

    rc_return Type;
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
                                           P->IsConst());

    } else if (const auto &D = llvm::dyn_cast<model::DefinedType>(&ModelType)) {
      rc_return fromTypeDefinition(D->unwrap(), RequireComplete, D->IsConst());

    } else if (const auto &A = llvm::dyn_cast<model::ArrayType>(&ModelType)) {
      rc_return make<clift::ArrayType>(rc_recur fromType(*A->ElementType(),
                                                         RequireComplete),
                                       A->ElementCount());

    } else if (const auto &P = llvm::dyn_cast<model::PointerType>(&ModelType)) {
      // If there's a pointer in the way, the base type does not have to be
      // complete.
      RequireComplete = false;

      rc_return make<clift::PointerType>(rc_recur fromType(*P->PointeeType(),
                                                           RequireComplete),
                                         P->PointerSize(),
                                         P->IsConst());

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
      if (auto RFT = llvm::dyn_cast<model::RawFunctionDefinition>(&ModelType)) {
        CompleteType = getRegisterSetType(*RFT);
      } else {
        CompleteType = fromTypeDefinition(ModelType, /*RequireComplete=*/true);
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
                       const model::TypeDefinition &ModelType,
                       const model::Binary &Binary) {
  return CliftConverter(Context, Binary, EmitError)
    .convertTypeDefinition(ModelType);
}

clift::ValueType
clift::importModelType(llvm::function_ref<mlir::InFlightDiagnostic()> EmitError,
                       mlir::MLIRContext &Context,
                       const model::Type &ModelType,
                       const model::Binary &Binary) {
  return CliftConverter(Context, Binary, EmitError).convertType(ModelType);
}
