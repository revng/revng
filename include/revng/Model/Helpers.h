#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <string>

#include "revng/ADT/CompilationTime.h"
#include "revng/ADT/Concepts.h"
#include "revng/Model/DynamicFunction.h"
#include "revng/Model/Function.h"
#include "revng/Model/Identifier.h"
#include "revng/Model/Segment.h"

namespace model {

template<typename Type>
concept EntityWithKey = requires(Type &&Value) {
  { Value.key() } -> std::convertible_to<const typename Type::Key &>;
};

template<typename Type>
concept EntityWithCustomName = requires(Type &&Value) {
  { Value.CustomName() } -> std::convertible_to<model::Identifier &>;
};

template<typename Type>
concept EntityWithOriginalName = requires(Type &&Value) {
  { Value.OriginalName() } -> std::convertible_to<std::string &>;
};

template<typename Type>
concept EntityWithComment = requires(Type &&Value) {
  { Value.Comment() } -> std::convertible_to<std::string &>;
};

template<typename Type>
concept EntityWithReturnValueComment = requires(Type &&Value) {
  { Value.ReturnValueComment() } -> std::convertible_to<std::string &>;
};

template<typename LHS, typename RHS>
LHS &copyMetadata(LHS &To, const RHS &From) {
  if constexpr (EntityWithCustomName<LHS> && EntityWithCustomName<RHS>)
    To.CustomName() = From.CustomName();

  if constexpr (EntityWithOriginalName<LHS> && EntityWithOriginalName<RHS>)
    To.OriginalName() = From.OriginalName();

  if constexpr (EntityWithComment<LHS> && EntityWithComment<RHS>)
    To.Comment() = From.Comment();

  if constexpr (EntityWithReturnValueComment<LHS>
                && EntityWithReturnValueComment<RHS>) {
    To.ReturnValueComment() = From.ReturnValueComment();
  }

  return To;
}

template<typename LHS, typename RHS>
LHS &moveMetadata(LHS &To, RHS &&From) {
  if constexpr (EntityWithCustomName<LHS> && EntityWithCustomName<RHS>)
    To.CustomName() = std::move(From.CustomName());

  if constexpr (EntityWithOriginalName<LHS> && EntityWithOriginalName<RHS>)
    To.OriginalName() = std::move(From.OriginalName());

  if constexpr (EntityWithComment<LHS> && EntityWithComment<RHS>)
    To.Comment() = std::move(From.Comment());

  if constexpr (EntityWithReturnValueComment<LHS>
                && EntityWithReturnValueComment<RHS>) {
    To.ReturnValueComment() = std::move(From.ReturnValueComment());
  }

  return To;
}

namespace editPath {

namespace detail {

template<typename Type, typename ValueType>
concept EntityWithValueType = requires(Type::value_type Value) {
  { static_cast<Type::value_type>(Value) } -> std::same_as<ValueType>;
};

template<typename Type, typename ValueType>
concept EntityWithUpcastableValueType = requires(Type::value_type Value) {
  {
    static_cast<Type::value_type>(Value)
  } -> std::same_as<UpcastablePointer<ValueType>>;
};

template<typename T, typename RootT>
struct RootContainerName {
private:
  using TLT = TupleLikeTraits<RootT>;
  static constexpr std::size_t Size = std::tuple_size_v<typename TLT::tuple>;
  static constexpr auto Index = compile_time::select<Size>([]<std::size_t I> {
    using Element = typename std::tuple_element_t<I, typename TLT::tuple>;
    return std::is_same_v<typename T::BaseClass, void> ?
             EntityWithValueType<Element, T> :
             EntityWithUpcastableValueType<Element, typename T::BaseClass>;
  });
  static_assert(Index.has_value());

public:
  static constexpr llvm::StringRef value = TLT::FieldNames[*Index];
};

template<typename RootT, EntityWithKey T>
inline std::string pathImpl(const T &Value, std::string &&FieldName) {
  return '/' + RootContainerName<T, RootT>::value.str() + '/'
         + getNameFromYAMLScalar(Value.key()) + '/' + std::move(FieldName);
}

// TODO: it might be worth it to reimplement this in terms of TupleTreePaths
//       instead of using string concatenation.
template<typename RootT, typename T, typename KeyT>
inline std::string pathImpl(const T &Value,
                            std::string &&FirstFieldName,
                            const KeyT &FieldKey,
                            std::string &&SecondFieldName) {
  return '/' + RootContainerName<T, RootT>::value.str() + '/'
         + getNameFromYAMLScalar(Value.key()) + '/' + std::move(FirstFieldName)
         + '/' + getNameFromYAMLScalar(FieldKey) + '/'
         + std::move(SecondFieldName);
}

} // namespace detail

template<EntityWithKey T>
inline std::string customName(const T &Value) {
  static_assert(EntityWithCustomName<T>);
  return detail::pathImpl<model::Binary, T>(Value, "CustomName");
}

inline std::string customName(const model::StructType &Struct,
                              const model::StructField::Key Field) {
  return detail::pathImpl<model::Binary>(Struct, "Fields", Field, "CustomName");
}
inline std::string customName(const model::StructType &Struct,
                              const model::StructField &Field) {
  return customName(Struct, Field.key());
}

inline std::string customName(const model::UnionType &Union,
                              const model::UnionField::Key Field) {
  return detail::pathImpl<model::Binary>(Union, "Fields", Field, "CustomName");
}
inline std::string customName(const model::UnionType &Union,
                              const model::UnionField &Field) {
  return customName(Union, Field.key());
}

inline std::string customName(const model::EnumType &Enum,
                              const model::EnumEntry::Key &Entry) {
  return detail::pathImpl<model::Binary>(Enum, "Entries", Entry, "CustomName");
}
inline std::string customName(const model::EnumType &Enum,
                              const model::EnumEntry &Entry) {
  return customName(Enum, Entry.key());
}

inline std::string customName(const model::CABIFunctionType &Function,
                              const model::Argument::Key &Argument) {
  return detail::pathImpl<model::Binary>(Function,
                                         "Arguments",
                                         Argument,
                                         "CustomName");
}
inline std::string customName(const model::CABIFunctionType &Function,
                              const model::Argument &Argument) {
  return customName(Function, Argument.key());
}

inline std::string customName(const model::RawFunctionType &Function,
                              const model::NamedTypedRegister::Key &Argument) {
  return detail::pathImpl<model::Binary>(Function,
                                         "Arguments",
                                         Argument,
                                         "CustomName");
}
inline std::string customName(const model::RawFunctionType &Function,
                              const model::NamedTypedRegister &Argument) {
  return customName(Function, Argument.key());
}

template<EntityWithKey T>
inline std::string comment(const T &Value) {
  static_assert(EntityWithComment<T>);
  return detail::pathImpl<model::Binary, T>(Value, "Comment");
}

inline std::string comment(const model::StructType &Struct,
                           const model::StructField::Key Field) {
  return detail::pathImpl<model::Binary>(Struct, "Fields", Field, "Comment");
}
inline std::string comment(const model::StructType &Struct,
                           const model::StructField &Field) {
  return comment(Struct, Field.key());
}

inline std::string comment(const model::UnionType &Union,
                           const model::UnionField::Key Field) {
  return detail::pathImpl<model::Binary>(Union, "Fields", Field, "Comment");
}
inline std::string comment(const model::UnionType &Union,
                           const model::UnionField &Field) {
  return comment(Union, Field.key());
}

inline std::string comment(const model::EnumType &Enum,
                           const model::EnumEntry::Key &Entry) {
  return detail::pathImpl<model::Binary>(Enum, "Entries", Entry, "Comment");
}
inline std::string comment(const model::EnumType &Enum,
                           const model::EnumEntry &Entry) {
  return comment(Enum, Entry.key());
}

inline std::string comment(const model::CABIFunctionType &Function,
                           const model::Argument::Key &Argument) {
  return detail::pathImpl<model::Binary>(Function,
                                         "Arguments",
                                         Argument,
                                         "Comment");
}
inline std::string comment(const model::CABIFunctionType &Function,
                           const model::Argument &Argument) {
  return comment(Function, Argument.key());
}

inline std::string comment(const model::RawFunctionType &Function,
                           const model::NamedTypedRegister::Key &Argument) {
  return detail::pathImpl<model::Binary>(Function,
                                         "Arguments",
                                         Argument,
                                         "Comment");
}
inline std::string comment(const model::RawFunctionType &Function,
                           const model::NamedTypedRegister &Argument) {
  return comment(Function, Argument.key());
}

inline std::string comment(const model::RawFunctionType &Function,
                           const model::TypedRegister::Key &ReturnValue) {
  return detail::pathImpl<model::Binary>(Function,
                                         "ReturnValues",
                                         ReturnValue,
                                         "Comment");
}
inline std::string comment(const model::RawFunctionType &Function,
                           const model::TypedRegister &ReturnValue) {
  return comment(Function, ReturnValue.key());
}

template<EntityWithKey T>
inline std::string returnValueComment(const T &Value) {
  static_assert(EntityWithReturnValueComment<T>);
  return detail::pathImpl<model::Binary, T>(Value, "ReturnValueComment");
}

} // namespace editPath

} // namespace model
