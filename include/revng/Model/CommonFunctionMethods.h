#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/Model/PrimitiveKind.h"

namespace model {

class TypeDefinition;
class RawFunctionDefinition;
class CABIFunctionDefinition;

template<typename CRTP>
class CommonFunctionMethods {
public:
  /// The helper for the prototype unwrapping.
  /// Use this when you need to access/modify the existing prototype,
  /// and \ref Prototype() when you need to assign a new one.
  model::TypeDefinition *prototype() {
    if (auto &This = *static_cast<CRTP *>(this); This.Prototype().isEmpty())
      return nullptr;
    else
      return &This.Prototype()->toPrototype();
  }

  /// The helper for the prototype unwrapping.
  /// Use this when you need to access/modify the existing prototype,
  /// and \ref Prototype() when you need to assign a new one.
  const model::TypeDefinition *prototype() const {
    if (auto &This = *static_cast<const CRTP *>(this);
        This.Prototype().isEmpty())
      return nullptr;
    else
      return &This.Prototype()->toPrototype();
  }

public:
  /// The helper for the prototype unwrapping.
  /// Use this when you need to access/modify the existing prototype,
  /// and \ref Prototype() when you need to assign a new one.
  model::RawFunctionDefinition *rawPrototype() {
    if (model::TypeDefinition *Prototype = prototype())
      return llvm::dyn_cast<model::RawFunctionDefinition>(Prototype);
    else
      return nullptr;
  }

  /// The helper for the prototype unwrapping.
  /// Use this when you need to access/modify the existing prototype,
  /// and \ref Prototype() when you need to assign a new one.
  const model::RawFunctionDefinition *rawPrototype() const {
    if (const model::TypeDefinition *Prototype = prototype())
      return llvm::dyn_cast<model::RawFunctionDefinition>(Prototype);
    else
      return nullptr;
  }

public:
  /// The helper for the prototype unwrapping.
  /// Use this when you need to access/modify the existing prototype,
  /// and \ref Prototype() when you need to assign a new one.
  model::CABIFunctionDefinition *cabiPrototype() {
    if (model::TypeDefinition *Prototype = prototype())
      return llvm::dyn_cast<model::CABIFunctionDefinition>(Prototype);
    else
      return nullptr;
  }

  /// The helper for the prototype unwrapping.
  /// Use this when you need to access/modify the existing prototype,
  /// and \ref Prototype() when you need to assign a new one.
  const model::CABIFunctionDefinition *cabiPrototype() const {
    if (const model::TypeDefinition *Prototype = prototype())
      return llvm::dyn_cast<model::CABIFunctionDefinition>(Prototype);
    else
      return nullptr;
  }
};

} // namespace model
