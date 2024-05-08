#pragma once

//
// This file is distributed under the MIT License. See LICENSE.mit for details.
//

#include "revng/Model/Identifier.h"
#include "revng/Model/NamedTypedRegister.h"
#include "revng/Model/Type.h"
#include "revng/Model/TypedRegister.h"

/* TUPLE-TREE-YAML
name: RawFunctionType
type: struct
inherits: Type
fields:
  - name: Arguments
    sequence:
      type: SortedVector
      elementType: NamedTypedRegister
  - name: ReturnValues
    sequence:
      type: SortedVector
      elementType: NamedTypedRegister
  - name: ReturnValueComment
    type: string
    optional: true
  - name: PreservedRegisters
    sequence:
      type: SortedVector
      elementType: Register
  - name: FinalStackOffset
    type: uint64_t
  - name: StackArgumentsType
    doc: The type of the struct representing stack arguments
    reference:
      pointeeType: Type
      rootType: Binary
    optional: true
TUPLE-TREE-YAML */

#include "revng/Model/Generated/Early/RawFunctionType.h"

class model::RawFunctionType : public model::generated::RawFunctionType {
public:
  static constexpr const char *AutomaticNamePrefix = "rawfunction_";

public:
  using generated::RawFunctionType::RawFunctionType;
  RawFunctionType() : generated::RawFunctionType(){};

public:
  Identifier name() const;

public:
  const llvm::SmallVector<model::QualifiedType, 4> edges() const {
    llvm::SmallVector<model::QualifiedType, 4> Result;

    for (auto &Argument : Arguments())
      Result.push_back(Argument.Type());
    for (auto &RV : ReturnValues())
      Result.push_back(RV.Type());
    if (not StackArgumentsType().empty())
      Result.push_back(QualifiedType{ StackArgumentsType(), {} });

    return Result;
  }

public:
  static bool classof(const Type *T) { return classof(T->key()); }
  static bool classof(const Key &K) { return std::get<1>(K) == AssociatedKind; }
};

#include "revng/Model/Generated/Late/RawFunctionType.h"
