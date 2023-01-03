#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <string>

#include "revng/ADT/Concepts.h"
#include "revng/Model/Identifier.h"

namespace model {

template<typename Type>
concept EntityWithMetadata = requires(Type &&Value) {
  { Value.CustomName() } -> convertible_to<model::Identifier &>;
  { Value.OriginalName() } -> convertible_to<std::string &>;
};

template<EntityWithMetadata LHS, EntityWithMetadata RHS>
LHS &copyMetadata(LHS &To, const RHS &From) {
  To.CustomName() = From.CustomName();
  To.OriginalName() = From.OriginalName();

  return To;
}

template<EntityWithMetadata LHS, EntityWithMetadata RHS>
LHS &moveMetadata(LHS &To, RHS &&From) {
  To.CustomName() = std::move(From.CustomName());
  To.OriginalName() = std::move(From.OriginalName());

  return To;
}

} // namespace model
