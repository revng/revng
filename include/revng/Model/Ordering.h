#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <concepts>
#include <map>

#include "revng/Model/TypeDefinition.h"

namespace model {

/// Orders pointers to type definitions by ID, rather than by the pointer.
///
/// Types are numbered as they are created, so a container that is iterated
/// while creating types cannot be keyed on something the allocator decides,
/// or the IDs it hands out change from one run to the next.
///
/// A null pointer compares less than any other one.
template<std::derived_from<TypeDefinition> T>
struct ByID {
  bool operator()(const T *LHS, const T *RHS) const {
    if (LHS == nullptr)
      return RHS != nullptr;

    if (RHS == nullptr)
      return false;

    return LHS->ID() < RHS->ID();
  }
};

/// A map keyed on type definitions, iterated in the order of their IDs
template<std::derived_from<TypeDefinition> Key, typename Value>
using TypeDefinitionMap = std::map<Key *, Value, ByID<Key>>;

} // namespace model
