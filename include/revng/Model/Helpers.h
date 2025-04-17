#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <string>

#include "revng/ADT/CompilationTime.h"
#include "revng/ADT/Concepts.h"
#include "revng/Model/DynamicFunction.h"
#include "revng/Model/Function.h"
#include "revng/Model/Segment.h"

//
// The following helpers are designed to help with handling metadata.
//
// Metadata are some common fields appearing in different (sometimes unrelated)
// types within the model. These helpers allow to work with such fields in
// a generic manner.

namespace model {

// Every supported metadata field should have an `EntityWithXXX` concept
// declared for it.

template<typename Type>
concept EntityWithKey = requires(const Type &Value) {
  { Value.key() } -> std::convertible_to<const typename Type::Key &>;
};

template<typename Type>
concept EntityWithName = requires(const Type &Value) {
  { Value.Name() } -> std::convertible_to<const std::string &>;
};

template<typename Type>
concept EntityWithComment = requires(const Type &Value) {
  { Value.Comment() } -> std::convertible_to<const std::string &>;
};

template<typename Type>
concept EntityWithReturnValueComment = requires(const Type &Value) {
  { Value.ReturnValueComment() } -> std::convertible_to<const std::string &>;
};

template<typename Type>
void ensureCompatibility() {
  static_assert(EntityWithName<Type> || EntityWithComment<Type>
                  || EntityWithReturnValueComment<Type>,
                "This would be a no-op.");
}

template<typename LHS, typename RHS>
LHS &copyMetadata(LHS &To, const RHS &From) {
  ensureCompatibility<LHS>();
  ensureCompatibility<RHS>();

  if constexpr (EntityWithName<LHS> && EntityWithName<RHS>)
    To.Name() = From.Name();

  if constexpr (EntityWithComment<LHS> && EntityWithComment<RHS>)
    To.Comment() = From.Comment();

  if constexpr (EntityWithReturnValueComment<LHS>
                && EntityWithReturnValueComment<RHS>) {
    To.ReturnValueComment() = From.ReturnValueComment();
  }

  return To;
}

template<typename LHS, typename RHS>
LHS &moveMetadata(LHS &To, const RHS &From) {
  ensureCompatibility<LHS>();
  ensureCompatibility<RHS>();

  if constexpr (EntityWithName<LHS> && EntityWithName<RHS>)
    To.Name() = std::move(From.Name());

  if constexpr (EntityWithComment<LHS> && EntityWithComment<RHS>)
    To.Comment() = std::move(From.Comment());

  if constexpr (EntityWithReturnValueComment<LHS>
                && EntityWithReturnValueComment<RHS>) {
    To.ReturnValueComment() = std::move(From.ReturnValueComment());
  }

  return To;
}

template<typename Type>
bool hasMetadata(const Type &Value) {
  ensureCompatibility<Type>();

  if constexpr (EntityWithName<Type>)
    if (!Value.Name().empty())
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
