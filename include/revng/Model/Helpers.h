#pragma once

//
// This file is distributed under the MIT License. See LICENSE.mit for details.
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

template<typename Type>
bool hasMetadata(Type &Value) {
  if constexpr (EntityWithCustomName<Type>)
    if (!Value.CustomName().empty())
      return true;

  if constexpr (EntityWithOriginalName<Type>)
    if (!Value.OriginalName().empty())
      return true;

  if constexpr (EntityWithComment<Type>)
    if (!Value.Comment().empty())
      return true;

  if constexpr (EntityWithReturnValueComment<Type>)
    if (!Value.ReturnValueComment().empty())
      return true;

  return false;
}

} // namespace model
