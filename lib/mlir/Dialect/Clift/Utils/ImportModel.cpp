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
  llvm::DenseMap<uint64_t, const model::Type *> IncompleteTypes;

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

  ~CliftConverter() {
    revng_assert(DefinitionGuardSet.empty());
    revng_assert(IncompleteTypes.empty());
  }

  clift::ValueType convertUnqualifiedType(const model::Type &ModelType) {
    const clift::ValueType T = getUnwrappedValueType(ModelType,
                                                     /*RequireComplete=*/true);
    if (T and not processIncompleteTypes()) {
      IncompleteTypes.clear();
      return nullptr;
    }
    return T;
  }

  clift::ValueType convertQualifiedType(const model::QualifiedType &ModelType) {
    const clift::ValueType T = getQualifiedValueType(ModelType,
                                                     /*RequireComplete=*/true);
    if (T and not processIncompleteTypes()) {
      IncompleteTypes.clear();
      return nullptr;
    }
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
    case model::PrimitiveTypeKind::Void:
      return clift::PrimitiveKind::VoidKind;
    case model::PrimitiveTypeKind::Generic:
      return clift::PrimitiveKind::GenericKind;
    case model::PrimitiveTypeKind::PointerOrNumber:
      return clift::PrimitiveKind::PointerOrNumberKind;
    case model::PrimitiveTypeKind::Number:
      return clift::PrimitiveKind::NumberKind;
    case model::PrimitiveTypeKind::Unsigned:
      return clift::PrimitiveKind::UnsignedKind;
    case model::PrimitiveTypeKind::Signed:
      return clift::PrimitiveKind::SignedKind;
    case model::PrimitiveTypeKind::Float:
      return clift::PrimitiveKind::FloatKind;

    case model::PrimitiveTypeKind::Invalid:
    case model::PrimitiveTypeKind::Count:
      revng_abort("These are invalid values. Something has gone wrong.");
    }
  }

  clift::ValueType getPrimitiveType(const model::PrimitiveType &ModelType,
                                    const bool Const) {
    return make<clift::PrimitiveType>(getPrimitiveKind(ModelType),
                                      ModelType.Size(),
                                      getBool(Const));
  }

  RecursiveCoroutine<clift::TypeDefinitionAttr>
  getTypeAttribute(const model::CABIFunctionType &ModelType) {
    RecursiveDefinitionGuard Guard(*this, ModelType.ID());
    if (not Guard) {
      if (EmitError)
        EmitError() << "Recursive definition of CABIFunctionType "
                    << ModelType.ID();
      rc_return nullptr;
    }

    AttributeVector<clift::FunctionArgumentAttr> Args;
    Args.reserve(ModelType.Arguments().size());

    for (const model::Argument &Argument : ModelType.Arguments()) {
      const auto Type = rc_recur getQualifiedValueType(Argument.Type());
      if (not Type)
        rc_return nullptr;
      const llvm::StringRef Name = Argument.name();
      const auto Attribute = make<clift::FunctionArgumentAttr>(Type, Name);
      if (not Attribute)
        rc_return nullptr;
      Args.push_back(Attribute);
    }

    const auto ReturnType = rc_recur getQualifiedValueType(ModelType
                                                             .ReturnType());
    if (not ReturnType)
      rc_return nullptr;

    rc_return make<clift::FunctionTypeAttr>(ModelType.ID(),
                                            ModelType.name(),
                                            ReturnType,
                                            Args);
  }

  RecursiveCoroutine<clift::TypeDefinitionAttr>
  getTypeAttribute(const model::EnumType &ModelType) {
    RecursiveDefinitionGuard Guard(*this, ModelType.ID());
    if (not Guard) {
      if (EmitError)
        EmitError() << "Recursive definition of EnumType " << ModelType.ID();
      rc_return nullptr;
    }

    const auto UnderlyingType = rc_recur
      getQualifiedValueType(ModelType.UnderlyingType());
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
  getScalarTupleType(const model::RawFunctionType &ModelType) {
    using ElementAttr = clift::ScalarTupleElementAttr;

    AttributeVector<ElementAttr> Elements;
    Elements.reserve(ModelType.ReturnValues().size());

    for (const model::NamedTypedRegister &Register : ModelType.ReturnValues()) {
      const auto RegisterType = rc_recur getQualifiedValueType(Register.Type());
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
  getTypeAttribute(const model::RawFunctionType &ModelType) {
    RecursiveDefinitionGuard Guard(*this, ModelType.ID());
    if (not Guard) {
      if (EmitError)
        EmitError() << "Recursive definition of RawFunctionType "
                    << ModelType.ID();
      rc_return nullptr;
    }

    clift::FunctionArgumentAttr StackArgument;
    size_t ArgumentsCount = 0;

    if (ModelType.StackArgumentsType().isValid()) {
      const auto Type = rc_recur
        getUnwrappedValueType(*ModelType.StackArgumentsType().get());
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
      const auto Type = rc_recur getQualifiedValueType(Register.Type());
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
      ReturnType = rc_recur
        getQualifiedValueType(ModelType.ReturnValues().begin()->Type());
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
  getTypeAttribute(const model::StructType &ModelType,
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
      const auto FieldType = rc_recur
        getQualifiedValueType(Field.Type(), /*RequireComplete=*/true);
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
  getTypeAttribute(const model::TypedefType &ModelType,
                   const bool RequireComplete) {
    std::optional<RecursiveDefinitionGuard> Guard;

    if (RequireComplete) {
      Guard.emplace(*this, ModelType.ID());
      if (not *Guard) {
        if (EmitError)
          EmitError() << "Recursive definition of TypedefType "
                      << ModelType.ID();
        rc_return nullptr;
      }
    }

    const auto UnderlyingType = rc_recur
      getQualifiedValueType(ModelType.UnderlyingType(), RequireComplete);
    if (not UnderlyingType)
      rc_return nullptr;
    rc_return make<clift::TypedefTypeAttr>(ModelType.ID(),
                                           ModelType.name(),
                                           UnderlyingType);
  }

  RecursiveCoroutine<clift::TypeDefinitionAttr>
  getTypeAttribute(const model::UnionType &ModelType,
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
      const auto FieldType = rc_recur
        getQualifiedValueType(Field.Type(), /*RequireComplete=*/true);
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
  getTypeAttribute(const model::Type &ModelType, bool &RequireComplete) {
    switch (ModelType.Kind()) {
    case model::TypeKind::CABIFunctionType: {
      RequireComplete = true;
      const auto &T = llvm::cast<model::CABIFunctionType>(ModelType);
      return getTypeAttribute(T);
    }
    case model::TypeKind::EnumType: {
      RequireComplete = true;
      const auto &T = llvm::cast<model::EnumType>(ModelType);
      return getTypeAttribute(T);
    }
    case model::TypeKind::RawFunctionType: {
      RequireComplete = true;
      const auto &T = llvm::cast<model::RawFunctionType>(ModelType);
      return getTypeAttribute(T);
    }
    case model::TypeKind::StructType: {
      const auto &T = llvm::cast<model::StructType>(ModelType);
      return getTypeAttribute(T, RequireComplete);
    }
    case model::TypeKind::TypedefType: {
      const auto &T = llvm::cast<model::TypedefType>(ModelType);
      return getTypeAttribute(T, RequireComplete);
    }
    case model::TypeKind::UnionType: {
      const auto &T = llvm::cast<model::UnionType>(ModelType);
      return getTypeAttribute(T, RequireComplete);
    }

    case model::TypeKind::PrimitiveType:
      revng_abort("Primitive types have no corresponding attribute.");

    case model::TypeKind::Invalid:
    case model::TypeKind::Count:
      revng_abort("These are invalid values. Something has gone wrong.");
    }
  }

  RecursiveCoroutine<clift::ValueType>
  getUnwrappedValueType(const model::Type &ModelType,
                        bool RequireComplete = false,
                        const bool Const = false) {
    if (const auto *const T = llvm::dyn_cast<model::PrimitiveType>(&ModelType))
      rc_return getPrimitiveType(*T, Const);

    const auto getDefinedType = [&](const auto Attr) -> clift::ValueType {
      return make<clift::DefinedType>(Attr, getBool(Const));
    };

    if (const auto It = Cache.find(ModelType.ID()); It != Cache.end())
      rc_return getDefinedType(It->second);

    const clift::TypeDefinitionAttr Attr = getTypeAttribute(ModelType,
                                                            RequireComplete);

    if (not Attr)
      rc_return nullptr;

    if (RequireComplete) {
      const auto R = Cache.try_emplace(ModelType.ID(), Attr);
      revng_assert(R.second);
    }

    rc_return getDefinedType(Attr);
  }

  RecursiveCoroutine<clift::ValueType>
  getQualifiedValueType(const model::QualifiedType &ModelType,
                        bool RequireComplete = false) {
    if (not ModelType.UnqualifiedType().isValid()) {
      if (EmitError)
        EmitError() << "Invalid UnqualifiedType in QualifiedType";
      rc_return nullptr;
    }

    auto Qualifiers = llvm::ArrayRef(ModelType.Qualifiers());

    // Qualifier::isPointer verifies itself, which interferes with unit testing.
    static constexpr auto IsPointer = [](const model::Qualifier &Q) {
      return Q.Kind() == model::QualifierKind::Pointer;
    };

    // If the set of qualifiers contains any pointers,
    // the base type does not need to be complete.
    if (RequireComplete and llvm::any_of(Qualifiers, IsPointer))
      RequireComplete = false;

    // Qualifier::isConst verifies itself, which interferes with unit testing.
    static constexpr auto IsConst = [](const model::Qualifier &Q) {
      return Q.Kind() == model::QualifierKind::Const;
    };

    const auto TakeConst = [&Qualifiers]() -> bool {
      if (not Qualifiers.empty() and IsConst(Qualifiers.back())) {
        Qualifiers = Qualifiers.slice(0, Qualifiers.size() - 1);
        return true;
      }
      return false;
    };

    clift::ValueType ResultType = rc_recur
      getUnwrappedValueType(*ModelType.UnqualifiedType().get(),
                            RequireComplete,
                            TakeConst());

    if (not ResultType)
      rc_return nullptr;

    const auto TakeQualifier = [&Qualifiers]() -> const model::Qualifier & {
      const model::Qualifier &Qualifier = Qualifiers.back();
      Qualifiers = Qualifiers.slice(0, Qualifiers.size() - 1);
      return Qualifier;
    };

    // Loop over (qualifier, const (optional)) pairs wrapping the type at each
    // iteration, until the list of qualifiers is exhausted.
    while (not Qualifiers.empty()) {
      switch (const model::Qualifier &Qualifier = TakeQualifier();
              Qualifier.Kind()) {
      case model::QualifierKind::Pointer:
        ResultType = make<clift::PointerType>(ResultType,
                                              Qualifier.Size(),
                                              getBool(TakeConst()));
        break;

      case model::QualifierKind::Array:
        ResultType = make<clift::ArrayType>(ResultType,
                                            Qualifier.Size(),
                                            getBool(TakeConst()));
        break;

      default:
        if (EmitError)
          EmitError() << "invalid type qualifiers";
        rc_return nullptr;
      }

      if (not ResultType)
        rc_return nullptr;
    }

    rc_return ResultType;
  }

  bool processIncompleteTypes() {
    while (not IncompleteTypes.empty()) {
      const auto Iterator = IncompleteTypes.begin();
      const model::Type &ModelType = *Iterator->second;
      IncompleteTypes.erase(Iterator);

      clift::ValueType CompleteType;
      if (const auto RFT = llvm::dyn_cast<model::RawFunctionType>(&ModelType)) {
        CompleteType = getScalarTupleType(*RFT);
      } else {
        CompleteType = getUnwrappedValueType(ModelType,
                                             /*RequireComplete=*/true);
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
                       const model::Type &ModelType) {
  return CliftConverter(Context, EmitError).convertUnqualifiedType(ModelType);
}

clift::ValueType
clift::importModelType(llvm::function_ref<mlir::InFlightDiagnostic()> EmitError,
                       mlir::MLIRContext &Context,
                       const model::QualifiedType &ModelType) {
  return CliftConverter(Context, EmitError).convertQualifiedType(ModelType);
}
