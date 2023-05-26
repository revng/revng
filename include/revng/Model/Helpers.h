#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <string>

#include "revng/ADT/Concepts.h"
#include "revng/Model/Identifier.h"

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

} // namespace model
