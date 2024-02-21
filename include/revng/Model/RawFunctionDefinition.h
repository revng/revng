#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/Model/Identifier.h"
#include "revng/Model/NamedTypedRegister.h"
#include "revng/Model/TypeDefinition.h"

/* TUPLE-TREE-YAML
name: RawFunctionDefinition
type: struct
inherits: TypeDefinition
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
      pointeeType: TypeDefinition
      rootType: Binary
    optional: true
TUPLE-TREE-YAML */

#include "revng/Model/Generated/Early/RawFunctionDefinition.h"

class model::RawFunctionDefinition
  : public model::generated::RawFunctionDefinition {
public:
  static constexpr const char *AutomaticNamePrefix = "rawfunction_";

public:
  using generated::RawFunctionDefinition::RawFunctionDefinition;

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
};

#include "revng/Model/Generated/Late/RawFunctionDefinition.h"
