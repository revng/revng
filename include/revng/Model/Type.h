#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <algorithm>
#include <cctype>
#include <compare>
#include <optional>
#include <string>

#include "revng/ADT/KeyedObjectTraits.h"
#include "revng/ADT/RecursiveCoroutine.h"
#include "revng/ADT/SortedVector.h"
#include "revng/ADT/UpcastablePointer.h"
#include "revng/ADT/UpcastablePointer/YAMLTraits.h"
#include "revng/Model/ABI.h"
#include "revng/Model/Register.h"
#include "revng/Support/Assert.h"
#include "revng/Support/Debug.h"
#include "revng/Support/YAMLTraits.h"
#include "revng/TupleTree/TupleTree.h"

namespace model {
class VerifyHelper;
class Binary;
class Type;
class PrimitiveType;
class EnumType;
class TypedefType;
class StructType;
class UnionType;
class CABIFunctionType;
class RawFunctionType;
class TypedRegister;
class NamedTypedRegister;

class Qualifier;
class QualifiedType;

class EnumEntry;
class AggregateField;
class StructField;
class UnionField;
class Argument;
class ABILocation;

} // end namespace model

template<typename T>
using Fields = typename TupleLikeTraits<T>::Fields;

namespace model {

extern const std::set<llvm::StringRef> ReservedKeywords;

/// \note Zero-sized identifiers are valid
class Identifier : public llvm::SmallString<16> {
public:
  using llvm::SmallString<16>::SmallString;
  using llvm::SmallString<16>::operator=;

public:
  static const Identifier Empty;

public:
  static Identifier fromString(llvm::StringRef Name) {
    revng_assert(not Name.empty());
    Identifier Result;

    // For reserved C keywords prepend a non-reserved prefix and we're done.
    if (ReservedKeywords.count(Name)) {
      Result += "prefix_";
      Result += Name;
      return Result;
    }

    const auto BecomesUnderscore = [](const char C) {
      return not std::isalnum(C) or C == '_';
    };

    // For invalid C identifiers prepend the our reserved prefix.
    if (std::isdigit(Name[0]) or BecomesUnderscore(Name[0]))
      Result += "prefix_";

    // Append the rest of the name
    Result += Name;

    // Convert all non-alphanumeric chars to underscores
    for (char &C : Result)
      if (not std::isalnum(C))
        C = '_';

    return Result;
  }

public:
  bool verify() const debug_function;
  bool verify(bool Assert) const debug_function;
  bool verify(VerifyHelper &VH) const;
};

} // namespace model

/// \brief KeyedObjectTraits for std::string based on its value
template<>
struct KeyedObjectTraits<model::Identifier>
  : public IdentityKeyedObjectTraits<model::Identifier> {};

template<>
struct llvm::yaml::ScalarTraits<model::Identifier> {
  static void
  output(const model::Identifier &Value, void *, llvm::raw_ostream &Output) {
    Output << Value;
  }

  static StringRef
  input(llvm::StringRef Scalar, void *, model::Identifier &Value) {
    Value = model::Identifier(Scalar);
    return StringRef();
  }

  static QuotingType mustQuote(StringRef) { return QuotingType::Double; }
};

namespace model::TypeKind {

/// \brief Enum for identifying different kind of model types.
enum Values {
  Invalid,
  Primitive,
  Enum,
  Typedef,
  Struct,
  Union,
  CABIFunctionType,
  RawFunctionType,
  Count,
};

inline llvm::StringRef getName(Values V) {
  switch (V) {
  case Invalid:
    return "Invalid";
  case Primitive:
    return "Primitive";
  case Enum:
    return "Enum";
  case Typedef:
    return "Typedef";
  case Struct:
    return "Struct";
  case Union:
    return "Union";
  case CABIFunctionType:
    return "CABIFunctionType";
  case RawFunctionType:
    return "RawFunctionType";
  default:
    revng_abort();
  }
}

inline model::TypeKind::Values fromName(llvm::StringRef Name) {
  if (Name == "Invalid") {
    return Invalid;
  } else if (Name == "Primitive") {
    return Primitive;
  } else if (Name == "Enum") {
    return Enum;
  } else if (Name == "Typedef") {
    return Typedef;
  } else if (Name == "Struct") {
    return Struct;
  } else if (Name == "Union") {
    return Union;
  } else if (Name == "CABIFunctionType") {
    return CABIFunctionType;
  } else if (Name == "RawFunctionType") {
    return RawFunctionType;
  } else {
    revng_abort();
  }
}

} // end namespace model::TypeKind

namespace llvm::yaml {

// Make model::TypeKind yaml-serializable, required for making model::Type
// yaml-serializable as well
template<>
struct ScalarEnumerationTraits<model::TypeKind::Values>
  : public NamedEnumScalarTraits<model::TypeKind::Values> {};

} // end namespace llvm::yaml

namespace model::QualifierKind {

/// \brief Enum for identifying different kinds of qualifiers.
//
// Notice that we are choosing to represent pointers and arrays as qualifiers.
// The idea is that a qualifier is something that you can add to a type T to
// obtain another type R, in such a way that if T is fully known also R is fully
// known. In this sense Pointer and Array types are qualified types.
enum Values { Invalid, Pointer, Array, Const, Count };

inline llvm::StringRef getName(Values V) {
  switch (V) {
  case Invalid:
    return "Invalid";
  case Pointer:
    return "Pointer";
  case Array:
    return "Array";
  case Const:
    return "Const";
  default:
    revng_abort();
  }
  revng_abort();
}

} // end namespace model::QualifierKind

namespace llvm::yaml {

// Make model::QualifierKind::Values yaml-serializable, required for making
// model::Qualifier yaml-serializable as well
template<>
struct ScalarEnumerationTraits<model::QualifierKind::Values>
  : public NamedEnumScalarTraits<model::QualifierKind::Values> {};

} // namespace llvm::yaml

// Make model::Type derived types usable with UpcastablePointer
template<>
struct concrete_types_traits<model::Type> {
  using type = std::tuple<model::PrimitiveType,
                          model::EnumType,
                          model::TypedefType,
                          model::StructType,
                          model::UnionType,
                          model::CABIFunctionType,
                          model::RawFunctionType>;
};

template<>
struct concrete_types_traits<const model::Type> {
  using type = std::tuple<const model::PrimitiveType,
                          const model::EnumType,
                          const model::TypedefType,
                          const model::StructType,
                          const model::UnionType,
                          const model::CABIFunctionType,
                          const model::RawFunctionType>;
};

/// \brief Concept to identify all types that are derived from model::Type
template<typename T>
concept IsModelType = std::is_base_of_v<model::Type, T>;

/// \brief Base class of model types used for LLVM-style RTTI
class model::Type {
public:
  using Key = std::pair<model::TypeKind::Values, uint64_t>;

public:
  TypeKind::Values Kind = TypeKind::Invalid;
  uint64_t ID = 0;

public:
  static bool classof(const Type *T) { return false; }

  Key key() const { return Key{ Kind, ID }; }

  Identifier name() const;

protected:
  Type(TypeKind::Values Kind, uint64_t ID) : Kind(Kind), ID(ID) {}
  Type(TypeKind::Values Kind);

public:
  bool operator==(const Type &Other) const { return key() == Other.key(); }

  bool operator<(const Type &Other) const { return key() < Other.key(); }

public:
  std::optional<uint64_t> size() const debug_function;
  RecursiveCoroutine<std::optional<uint64_t>> size(VerifyHelper &VH) const;

public:
  bool verify() const debug_function;
  bool verify(bool Assert) const debug_function;
  RecursiveCoroutine<bool> verify(VerifyHelper &VH) const;
  void dump() const debug_function;

protected:
  bool verifyBase(VerifyHelper &VH) const;
};

INTROSPECTION_NS(model, Type, Kind, ID);

template<>
struct llvm::yaml::MappingTraits<model::Type>
  : public TupleLikeMappingTraits<model::Type> {};

static_assert(Yamlizable<model::Type>);

namespace model {

using UpcastableType = UpcastablePointer<model::Type>;

template<size_t I = 0>
inline model::UpcastableType
makeTypeWithID(model::TypeKind::Values Kind, uint64_t ID);

} // end namespace model

template<>
struct llvm::yaml::ScalarTraits<model::Type::Key>
  : CompositeScalar<model::Type::Key, '-'> {};

// Make UpcastablePointers to model types usable in KeyedObject containers
template<>
struct KeyedObjectTraits<model::UpcastableType> {
  static model::Type::Key key(const model::UpcastableType &Val) {
    return Val->key();
  }
  static model::UpcastableType fromKey(const model::Type::Key &K) {
    return model::makeTypeWithID(K.first, K.second);
  }
};

/// \brief Make UpcastableType yaml-serializable polymorphically
template<>
struct llvm::yaml::MappingTraits<model::UpcastableType>
  : public PolymorphicMappingTraits<model::UpcastableType> {};

/// \brief A qualifier for a model::Type
class model::Qualifier {
public:
  QualifierKind::Values Kind = QualifierKind::Invalid;
  /// Size: size in bytes for Pointer, number of elements for Array, 0 otherwise
  uint64_t Size = 0;

public:
  // Kind is not Invalid, Pointer and Const have no Size, Array has Size.
  bool verify() const debug_function;
  bool verify(bool Assert) const debug_function;
  bool verify(VerifyHelper &VH) const;
  void dump() const debug_function;

public:
  static Qualifier createConst() { return { QualifierKind::Const, 0 }; }

  static Qualifier createPointer(uint64_t Size) {
    return { QualifierKind::Pointer, Size };
  }

  static Qualifier createArray(uint64_t S) {
    return { QualifierKind::Array, S };
  }

public:
  bool isConstQualifier() const {
    revng_assert(verify(true));
    return Kind == QualifierKind::Const;
  }

  bool isArrayQualifier() const {
    revng_assert(verify(true));
    return Kind == QualifierKind::Array;
  }

  bool isPointerQualifier() const {
    revng_assert(verify(true));
    return Kind == QualifierKind::Pointer;
  }

  std::strong_ordering operator<=>(const Qualifier &) const = default;
};
INTROSPECTION_NS(model, Qualifier, Kind, Size);

/// \brief Make Qualifier yaml-serializable
template<>
struct llvm::yaml::MappingTraits<model::Qualifier>
  : public TupleLikeMappingTraits<model::Qualifier,
                                  Fields<model::Qualifier>::Size> {};

// Make std::vector<model::Qualifier> yaml-serializable as a sequence
LLVM_YAML_IS_SEQUENCE_VECTOR(model::Qualifier)

namespace model {

using TypePath = TupleTreeReference<model::Type, model::Binary>;

} // end namespace model

namespace model::PrimitiveTypeKind {

// WARNING: these end up in type IDs, changing these means breaks the file
// format
enum Values {
  Invalid,
  Void,
  Generic,
  PointerOrNumber,
  Number,
  Unsigned,
  Signed,
  Float,
  Count
};

inline llvm::StringRef getName(Values V) {
  switch (V) {
  case Invalid:
    return "Invalid";
  case Void:
    return "Void";
  case Generic:
    return "Generic";
  case PointerOrNumber:
    return "PointerOrNumber";
  case Number:
    return "Number";
  case Unsigned:
    return "Unsigned";
  case Signed:
    return "Signed";
  case Float:
    return "Float";
  default:
    revng_abort();
  }
}

inline model::PrimitiveTypeKind::Values fromName(llvm::StringRef Name) {
  if (Name == "Invalid") {
    return Invalid;
  } else if (Name == "Void") {
    return Void;
  } else if (Name == "Generic") {
    return Generic;
  } else if (Name == "PointerOrNumber") {
    return PointerOrNumber;
  } else if (Name == "Number") {
    return Number;
  } else if (Name == "Unsigned") {
    return Unsigned;
  } else if (Name == "Signed") {
    return Signed;
  } else if (Name == "Float") {
    return Float;
  } else {
    revng_abort();
  }
}

} // end namespace model::PrimitiveTypeKind

namespace llvm::yaml {

// Make model::PrimitiveTypeKind::Values yaml-serializable
template<>
struct ScalarEnumerationTraits<model::PrimitiveTypeKind::Values>
  : public NamedEnumScalarTraits<model::PrimitiveTypeKind::Values> {};

} // end namespace llvm::yaml

/// \brief A qualified version of a model::Type. Can have many nested qualifiers
class model::QualifiedType {
public:
  TypePath UnqualifiedType;
  std::vector<Qualifier> Qualifiers = {};

public:
  bool operator==(const model::QualifiedType &Other) const = default;

public:
  std::optional<uint64_t> size() const debug_function;
  RecursiveCoroutine<std::optional<uint64_t>> size(VerifyHelper &VH) const;

  bool isScalar() const;
  bool isPrimitive(model::PrimitiveTypeKind::Values V) const;
  bool isVoid() const { return isPrimitive(model::PrimitiveTypeKind::Void); }
  bool isFloat() const { return isPrimitive(model::PrimitiveTypeKind::Float); }

public:
  bool verify() const debug_function;
  bool verify(bool Assert) const debug_function;
  RecursiveCoroutine<bool> verify(VerifyHelper &VH) const;
  void dump() const debug_function;
};
INTROSPECTION_NS(model, QualifiedType, UnqualifiedType, Qualifiers);

/// \brief Make QualifiedType yaml-serializable
template<>
struct llvm::yaml::MappingTraits<model::QualifiedType>
  : public TupleLikeMappingTraits<model::QualifiedType,
                                  Fields<model::QualifiedType>::Qualifiers> {};

/// \brief A primitive type in model: sized integers, booleans, floats and void.
class model::PrimitiveType : public model::Type {
public:
  static constexpr const char *Tag = "!Primitive";
  static constexpr const TypeKind::Values AssociatedKind = TypeKind::Primitive;

public:
  PrimitiveTypeKind::Values PrimitiveKind = model::PrimitiveTypeKind::Invalid;
  /// Size in bytes
  uint8_t Size = 0;

public:
  PrimitiveType(PrimitiveTypeKind::Values PrimitiveKind, uint8_t ByteSize);
  PrimitiveType(uint64_t ID);
  PrimitiveType() : PrimitiveType(model::PrimitiveTypeKind::Void) {}

public:
  Identifier name() const;

public:
  static bool classof(const Type *T) { return T->Kind == TypeKind::Primitive; }
  bool operator==(const PrimitiveType &Other) const = default;
};
INTROSPECTION_NS(model, PrimitiveType, Kind, ID, PrimitiveKind, Size);

template<>
struct llvm::yaml::MappingTraits<model::PrimitiveType>
  : public TupleLikeMappingTraits<model::PrimitiveType> {};

/// \brief An entry in a model enum, with a name and a value.
class model::EnumEntry {
public:
  uint64_t Value;
  Identifier CustomName;
  SortedVector<Identifier> Aliases;

public:
  EnumEntry(uint64_t Value) : Value(Value) {}
  EnumEntry() : EnumEntry(0) {}

public:
  bool operator==(const EnumEntry &) const = default;

  // The entry should have a non-empty name, the name should not be a valid
  // alias, and there should not be empty aliases.
  bool verify() const debug_function;
  bool verify(bool Assert) const debug_function;
  bool verify(VerifyHelper &VH) const;
  void dump() const debug_function;
};
INTROSPECTION_NS(model, EnumEntry, Value, CustomName, Aliases);

/// \brief KeyedObjectTraits for model::EnumEntry based on its value
template<>
struct KeyedObjectTraits<model::EnumEntry> {

  static uint64_t key(const model::EnumEntry &E) { return E.Value; }

  static model::EnumEntry fromKey(uint64_t V) { return model::EnumEntry{ V }; }
};

/// \brief Make EnumEntry yaml-serializable
template<>
struct llvm::yaml::MappingTraits<model::EnumEntry>
  : public TupleLikeMappingTraits<model::EnumEntry,
                                  Fields<model::EnumEntry>::CustomName> {};

/// \brief An enum type in model. Enums are actually typedefs of unnamed
/// enums.
class model::EnumType : public model::Type {
public:
  static constexpr const char *Tag = "!Enum";
  static constexpr const char *AutomaticNamePrefix = "enum_";
  static constexpr const TypeKind::Values AssociatedKind = TypeKind::Enum;

public:
  Identifier CustomName;
  // TODO: once we decide to embed path prefixes into TupleTreeReference, it'll
  //       be easier to restrict this TupleTreeReference to point to a
  //       PrimitiveType. For now, it's not really worth the extra complexity.
  TypePath UnderlyingType;
  SortedVector<EnumEntry> Entries;

public:
  /// \note Not to be used directly, only KeyedObjectTraits should use this
  EnumType(uint64_t ID) : Type(AssociatedKind, ID) {}
  EnumType() : Type(AssociatedKind) {}

public:
  Identifier name() const;
  static bool classof(const Type *T) { return T->Kind == TypeKind::Enum; }
  bool operator==(const EnumType &Other) const = default;
};
INTROSPECTION_NS(model,
                 EnumType,
                 Kind,
                 ID,
                 CustomName,
                 UnderlyingType,
                 Entries);

template<>
struct llvm::yaml::MappingTraits<model::EnumType>
  : public TupleLikeMappingTraits<model::EnumType,
                                  Fields<model::EnumType>::CustomName> {};

/// \brief A typedef type in model.
class model::TypedefType : public model::Type {
public:
  static constexpr const char *Tag = "!Typedef";
  static constexpr const char *AutomaticNamePrefix = "typedef_";
  static constexpr const TypeKind::Values AssociatedKind = TypeKind::Typedef;

public:
  Identifier CustomName;
  QualifiedType UnderlyingType;

public:
  /// \note Not to be used directly, only KeyedObjectTraits should use this
  TypedefType(uint64_t ID) : Type(AssociatedKind, ID) {}
  TypedefType() : Type(AssociatedKind) {}

public:
  Identifier name() const;
  static bool classof(const Type *T) { return T->Kind == TypeKind::Typedef; }
};
INTROSPECTION_NS(model, TypedefType, Kind, ID, CustomName, UnderlyingType);

template<>
struct llvm::yaml::MappingTraits<model::TypedefType>
  : public TupleLikeMappingTraits<model::TypedefType,
                                  Fields<model::TypedefType>::CustomName> {};

/// \brief A field of an aggregate type in model, with qualified type and name
class model::AggregateField {
public:
  Identifier CustomName;
  QualifiedType Type;

public:
  AggregateField() {}

  bool operator==(const AggregateField &) const = default;

public:
  bool verify() const debug_function;
  bool verify(bool Assert) const debug_function;
  RecursiveCoroutine<bool> verify(VerifyHelper &VH) const;
};

/// \brief A field of a struct type in model, with offset, qualified type, and
/// name
class model::StructField : public model::AggregateField {
public:
  uint64_t Offset = 0;

public:
  StructField(uint64_t Offset) : Offset(Offset) {}
  StructField() : StructField(0) {}

  Identifier name() const;

  bool operator==(const StructField &) const = default;
};
INTROSPECTION_NS(model, StructField, CustomName, Type, Offset);

/// \brief KeyedObjectTraits for model::StructType based on its byte-offset in
/// struct
template<>
struct KeyedObjectTraits<model::StructField> {

  static uint64_t key(const model::StructField &Val) { return Val.Offset; }

  static model::StructField fromKey(const uint64_t &Offset) {
    return model::StructField(Offset);
  }
};

/// \brief Make StructField yaml-serializable
template<>
struct llvm::yaml::MappingTraits<model::StructField>
  : public TupleLikeMappingTraits<model::StructField,
                                  Fields<model::StructField>::CustomName> {};

/// \brief A struct type in model. Structs are actually typedefs of unnamed
/// structs in C.
class model::StructType : public model::Type {
public:
  static constexpr const char *Tag = "!Struct";
  static constexpr const char *AutomaticNamePrefix = "struct_";
  static constexpr const TypeKind::Values AssociatedKind = TypeKind::Struct;

public:
  Identifier CustomName;
  SortedVector<StructField> Fields;
  /// Size in bytes
  uint64_t Size = 0;

public:
  /// \note Not to be used directly, only KeyedObjectTraits should use this
  StructType(uint64_t ID) : Type(AssociatedKind, ID) {}
  StructType() : Type(AssociatedKind) {}

public:
  Identifier name() const;
  static bool classof(const Type *T) { return T->Kind == TypeKind::Struct; }
};
INTROSPECTION_NS(model, StructType, Kind, ID, CustomName, Fields, Size);

template<>
struct llvm::yaml::MappingTraits<model::StructType>
  : public TupleLikeMappingTraits<model::StructType,
                                  Fields<model::StructType>::CustomName> {};

/// \brief A field of a union type in model, with position, qualified type, and
/// name
class model::UnionField : public model::AggregateField {
public:
  uint64_t Index;

public:
  UnionField(uint64_t Index) : AggregateField(), Index(Index) {}
  UnionField() : UnionField(0) {}

public:
  Identifier name() const;

  bool operator==(const UnionField &Other) const = default;
};
INTROSPECTION_NS(model, UnionField, CustomName, Type, Index);

/// \brief KeyedObjectTraits for model::UnionField based on its position in the
/// union
template<>
struct KeyedObjectTraits<model::UnionField> {

  static uint64_t key(const model::UnionField &Val) { return Val.Index; }

  static model::UnionField fromKey(const uint64_t &Index) {
    return model::UnionField(Index);
  }
};

/// \brief Make UnionField yaml-serializable
template<>
struct llvm::yaml::MappingTraits<model::UnionField>
  : public TupleLikeMappingTraits<model::UnionField,
                                  Fields<model::UnionField>::CustomName> {};

/// \brief A union type in model. Unions are actually typedefs of unnamed
/// unions in C.
class model::UnionType : public model::Type {
public:
  static constexpr const char *Tag = "!Union";
  static constexpr const char *AutomaticNamePrefix = "union_";
  static constexpr const TypeKind::Values AssociatedKind = TypeKind::Union;

public:
  Identifier CustomName;
  SortedVector<UnionField> Fields;

public:
  /// \note Not to be used directly, only KeyedObjectTraits should use this
  UnionType(uint64_t ID) : Type(AssociatedKind, ID) {}
  UnionType() : Type(AssociatedKind) {}

public:
  Identifier name() const;
  static bool classof(const Type *T) { return T->Kind == TypeKind::Union; }
};
INTROSPECTION_NS(model, UnionType, Kind, ID, CustomName, Fields);

template<>
struct llvm::yaml::MappingTraits<model::UnionType>
  : public TupleLikeMappingTraits<model::UnionType,
                                  Fields<model::UnionType>::CustomName> {};

/// \brief
class model::TypedRegister {
public:
  Register::Values Location;
  QualifiedType Type;

public:
  TypedRegister(Register::Values Location) : Location(Location) {}
  TypedRegister() : TypedRegister(Register::Invalid) {}

public:
  bool operator==(const TypedRegister &) const = default;

public:
  bool verify() const debug_function;
  bool verify(bool Assert) const debug_function;
  RecursiveCoroutine<bool> verify(VerifyHelper &VH) const;
  void dump() const debug_function;
};
INTROSPECTION_NS(model, TypedRegister, Location, Type);

template<>
struct KeyedObjectTraits<model::TypedRegister> {
  static model::Register::Values key(const model::TypedRegister &Obj) {
    return Obj.Location;
  }

  static model::TypedRegister fromKey(const model::Register::Values &Register) {
    return model::TypedRegister(Register);
  }
};

template<>
struct llvm::yaml::MappingTraits<model::TypedRegister>
  : public TupleLikeMappingTraits<model::TypedRegister> {};

/// \brief
class model::NamedTypedRegister : public TypedRegister {
public:
  Identifier CustomName;

public:
  using TypedRegister::TypedRegister;
  using TypedRegister::operator==;

public:
  Identifier name() const;

public:
  bool verify() const debug_function;
  bool verify(bool Assert) const debug_function;
  RecursiveCoroutine<bool> verify(VerifyHelper &VH) const;
  void dump() const debug_function;
};
INTROSPECTION_NS(model, NamedTypedRegister, Location, Type, CustomName);

template<>
struct KeyedObjectTraits<model::NamedTypedRegister> {
  static model::Register::Values key(const model::NamedTypedRegister &Obj) {
    return Obj.Location;
  }

  static model::NamedTypedRegister
  fromKey(const model::Register::Values &Register) {
    return model::NamedTypedRegister(Register);
  }
};

constexpr auto NTRCustomName = Fields<model::NamedTypedRegister>::CustomName;

template<>
struct llvm::yaml::MappingTraits<model::NamedTypedRegister>
  : public TupleLikeMappingTraits<model::NamedTypedRegister, NTRCustomName> {};

class model::RawFunctionType : public model::Type {
public:
  static constexpr const char *Tag = "!RawFunctionType";
  static constexpr const char *AutomaticNamePrefix = "rawfunction_";
  static constexpr const auto AssociatedKind = TypeKind::RawFunctionType;

public:
  Identifier CustomName;
  SortedVector<NamedTypedRegister> Arguments;
  SortedVector<TypedRegister> ReturnValues;
  SortedVector<Register::Values> PreservedRegisters;
  uint64_t FinalStackOffset = 0;
  TypePath StackArgumentsType;

public:
  /// \note Not to be used directly, only KeyedObjectTraits should use this
  RawFunctionType(uint64_t ID) : Type(AssociatedKind, ID) {}
  RawFunctionType() : Type(AssociatedKind) {}

public:
  Identifier name() const;
  static bool classof(const Type *T) {
    return T->Kind == TypeKind::RawFunctionType;
  }
};
INTROSPECTION_NS(model,
                 RawFunctionType,
                 Kind,
                 ID,
                 CustomName,
                 Arguments,
                 ReturnValues,
                 PreservedRegisters,
                 FinalStackOffset,
                 StackArgumentsType);

using RawFunctionTypeFields = Fields<model::RawFunctionType>;

template<>
struct llvm::yaml::MappingTraits<model::RawFunctionType>
  : public TupleLikeMappingTraits<model::RawFunctionType,
                                  RawFunctionTypeFields::CustomName,
                                  RawFunctionTypeFields::StackArgumentsType> {};

template<typename K, typename V>
V getOrDefault(const std::map<K, V> &Map, const K &Key, const V &Default) {
  auto It = Map.find(Key);
  if (It == Map.end())
    return Default;
  else
    return It->second;
}

/// \brief The argument of a function type
///
/// It features an argument index (the key), a type and an optional name
class model::Argument {
public:
  uint64_t Index;
  QualifiedType Type;
  Identifier CustomName;

public:
  Argument(uint64_t Index) : Index(Index) {}
  Argument() : Argument(0) {}

public:
  bool operator==(const Argument &) const = default;

  Identifier name() const;

public:
  bool verify() const debug_function;
  bool verify(bool Assert) const debug_function;
  RecursiveCoroutine<bool> verify(VerifyHelper &VH) const;
  void dump() const debug_function;
};
INTROSPECTION_NS(model, Argument, Index, Type, CustomName);

/// \brief KeyedObjectTraits for model::Argument based on its position
template<>
struct KeyedObjectTraits<model::Argument> {

  static uint64_t key(const model::Argument &Val) { return Val.Index; }

  static model::Argument fromKey(const uint64_t &Index) {
    return model::Argument(Index);
  }
};

/// \brief Make Argument yaml-serializable
template<>
struct llvm::yaml::MappingTraits<model::Argument>
  : public TupleLikeMappingTraits<model::Argument,
                                  Fields<model::Argument>::CustomName> {};

/// \brief The function type described through a C-like prototype plus an ABI
///
/// This is an "high level" representation of the prototype of a function. It is
/// expressed as list of arguments composed by an index and a type. No
/// information about the register is embedded. That information is implicit in
/// the ABI this type is associated to.
///
/// \see RawFunctionType
class model::CABIFunctionType : public model::Type {
public:
  static constexpr const char *Tag = "!CABIFunctionType";
  static constexpr const char *AutomaticNamePrefix = "cabifunction_";
  static constexpr const auto AssociatedKind = TypeKind::CABIFunctionType;

  Identifier CustomName;
  abi::Values ABI = abi::Invalid;
  QualifiedType ReturnType;
  SortedVector<Argument> Arguments;
  // TODO: handle variadic functions

public:
  /// \note Not to be used directly, only KeyedObjectTraits should use this
  CABIFunctionType(uint64_t ID) : Type(AssociatedKind, ID) {}
  CABIFunctionType() : Type(AssociatedKind) {}

public:
  Identifier name() const;
  static bool classof(const Type *T) {
    return T->Kind == TypeKind::CABIFunctionType;
  }
};
INTROSPECTION_NS(model,
                 CABIFunctionType,
                 Kind,
                 ID,
                 CustomName,
                 ABI,
                 ReturnType,
                 Arguments);

template<>
struct llvm::yaml::MappingTraits<model::CABIFunctionType>
  : public TupleLikeMappingTraits<model::CABIFunctionType,
                                  Fields<model::CABIFunctionType>::CustomName> {
};

namespace model {

template<size_t I>
inline model::UpcastableType
makeTypeWithID(model::TypeKind::Values K, uint64_t ID) {
  using concrete_types = concrete_types_traits_t<model::Type>;
  if constexpr (I < std::tuple_size_v<concrete_types>) {
    using type = std::tuple_element_t<I, concrete_types>;
    if (type::AssociatedKind == K)
      return model::UpcastableType(new type(ID));
    else
      return model::makeTypeWithID<I + 1>(K, ID);
  } else {
    return model::UpcastableType(nullptr);
  }
}

using TypesSet = SortedVector<UpcastableType>;

} // end namespace model

static_assert(validateTupleTree<model::TypesSet>(IsYamlizable),
              "All elements of the model type system must be YAMLizable");

namespace model {

template<IsModelType T, typename... Args>
inline UpcastableType makeType(Args &&...A) {
  return UpcastableType::make<T>(std::forward<Args>(A)...);
}

} // end namespace model
