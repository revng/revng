#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <string>

#include "revng/ADT/Concepts.h"
#include "revng/Model/DynamicFunction.h"
#include "revng/Model/Function.h"
#include "revng/Model/Identifier.h"
#include "revng/Model/Segment.h"

namespace model {

template<typename Type>
concept EntityWithCustomName = requires(Type &&Value) {
  { Value.CustomName() } -> convertible_to<model::Identifier &>;
};

template<typename Type>
concept EntityWithOriginalName = requires(Type &&Value) {
  { Value.OriginalName() } -> convertible_to<std::string &>;
};

template<typename Type>
concept EntityWithComment = requires(Type &&Value) {
  { Value.Comment() } -> convertible_to<std::string &>;
};

template<typename Type>
concept EntityWithReturnValueComment = requires(Type &&Value) {
  { Value.ReturnValueComment() } -> convertible_to<std::string &>;
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

template<typename T>
inline std::string str(const T &Obj) {
  return getNameFromYAMLScalar(Obj);
};

inline std::string getCustomNamePath(const model::Segment &Segment) {
  return "/Segments/" + str(Segment.key()) + "/CustomName";
}

inline std::string getCustomNamePath(const model::DynamicFunction &DF) {
  return "/ImportedDynamicFunctions/" + str(DF.key()) + "/CustomName";
}

inline std::string getCustomNamePath(const model::Function &Function) {
  return "/Functions/" + str(Function.key()) + "/CustomName";
}

inline std::string getCustomNamePath(const model::Type &Type) {
  return "/Types/" + str(Type.key()) + "/CustomName";
}

template<typename T>
  requires std::same_as<model::UnionType, T>
           or std::same_as<model::StructType, T>
inline std::string getCustomNamePath(const T &Obj, uint64_t FieldIdx) {
  return "/Types/" + str(Obj.key()) + "/Fields/" + std::to_string(FieldIdx)
         + "/CustomName";
}

inline std::string getCustomNamePath(const model::StructType &Struct,
                                     const model::StructField Field) {
  return "/Types/" + str(Struct.key()) + "/Fields/" + str(Field.key())
         + "/CustomName";
}

inline std::string getCustomNamePath(const model::UnionType &Union,
                                     const model::UnionField Field) {
  return "/Types/" + str(Union.key()) + "/Fields/" + str(Field.key())
         + "/CustomName";
}

inline std::string getCustomNamePath(const model::EnumType &Type,
                                     const model::EnumEntry &Entry) {
  return "/Types/" + str(Type.key()) + "/" + str(Entry.key()) + "/CustomName";
}

inline std::string getCustomNamePath(const model::CABIFunctionType &Function,
                                     const model::Argument &Argument) {
  return "/Types/" + str(Function.key()) + "/" + str(Argument.key())
         + "/CustomName";
}

inline std::string getCustomNamePath(const model::RawFunctionType &Function,
                                     const model::NamedTypedRegister &NTR) {
  return "/Types/" + str(Function.key()) + "/" + str(NTR.key()) + "/CustomName";
}

} // namespace editPath

} // namespace model
