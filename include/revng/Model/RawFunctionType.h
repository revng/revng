#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/Model/Identifier.h"
#include "revng/Model/NamedTypedRegister.h"
#include "revng/Model/Type.h"
#include "revng/Model/TypeKind.h"
#include "revng/Model/TypedRegister.h"

/* TUPLE-TREE-YAML
name: RawFunctionType
type: struct
inherits: Type
fields:
  - name: Arguments
    sequence:
      type: SortedVector
      elementType: model::NamedTypedRegister
  - name: ReturnValues
    sequence:
      type: SortedVector
      elementType: model::TypedRegister
  - name: PreservedRegisters
    sequence:
      type: SortedVector
      elementType: model::Register::Values
  - name: FinalStackOffset
    type: uint64_t
  - name: StackArgumentsType
    reference:
      pointeeType: model::Type
      rootType: model::Binary
    optional: true
TUPLE-TREE-YAML */

#include "revng/Model/Generated/Early/RawFunctionType.h"

class model::RawFunctionType : public model::generated::RawFunctionType {
public:
  static constexpr const char *AutomaticNamePrefix = "rawfunction_";
  static constexpr const TypeKind::Values
    AssociatedKind = TypeKind::RawFunctionType;

public:
  using generated::RawFunctionType::RawFunctionType;
  RawFunctionType() : generated::RawFunctionType() { Kind = AssociatedKind; };

public:
  Identifier name() const;

public:
  llvm::SmallVector<model::QualifiedType, 4> edges() {
    llvm::SmallVector<model::QualifiedType, 4> Result;

    for (auto &Argument : Arguments)
      Result.push_back(Argument.Type);
    for (auto &RV : ReturnValues)
      Result.push_back(RV.Type);

    return Result;
  }

public:
  static bool classof(const Type *T) { return classof(T->key()); }
  static bool classof(const Key &K) { return std::get<0>(K) == AssociatedKind; }
};

#include "revng/Model/Generated/Late/RawFunctionType.h"
