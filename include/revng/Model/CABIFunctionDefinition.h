#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/ADT/SortedVector.h"
#include "revng/Model/Argument.h"
#include "revng/Model/Identifier.h"
#include "revng/Model/Type.h"
#include "revng/Model/TypeDefinition.h"

/* TUPLE-TREE-YAML
name: CABIFunctionDefinition
doc: |
  The function type described through a C-like prototype plus an ABI.

  This is an "high level" representation of the prototype of a function. It is
  expressed as list of arguments composed by an index and a type. No
  information about the register is embedded. That information is implicit in
  the ABI this type is associated to.

  In contrast, a `RawFunctionType` is not associated to any ABI and explicitly
  describes, among other things, what registers are used to pass arguments and
  return values.
type: struct
inherits: TypeDefinition
fields:
  - name: ABI
    type: ABI
    doc: The C ABI associated to this function type.
  - name: ReturnType
    type: Type
    doc: The function return type.
    optional: true
    upcastable: true
  - name: ReturnValueComment
    type: string
    optional: true
  - name: Arguments
    doc: The list of formal arguments of the function type.
    sequence:
      type: SortedVector
      elementType: Argument
TUPLE-TREE-YAML */

#include "revng/Model/Generated/Early/CABIFunctionDefinition.h"

class model::CABIFunctionDefinition
  : public model::generated::CABIFunctionDefinition {
public:
  using generated::CABIFunctionDefinition::CABIFunctionDefinition;

  Argument &addArgument(UpcastableType &&Type) {
    auto &&[Iterator, Success] = Arguments().emplace(Arguments().size());
    revng_assert(Success);
    Iterator->Type() = std::move(Type);
    return *Iterator;
  }

public:
  llvm::SmallVector<const model::Type *, 4> edges() const {
    llvm::SmallVector<const model::Type *, 4> Result;

    for (const model::Argument &Argument : Arguments())
      if (!Argument.Type().isEmpty())
        Result.push_back(Argument.Type().get());

    if (!ReturnType().isEmpty())
      Result.push_back(ReturnType().get());

    return Result;
  }
  llvm::SmallVector<model::Type *, 4> edges() {
    llvm::SmallVector<model::Type *, 4> Result;

    for (model::Argument &Argument : Arguments())
      if (!Argument.Type().isEmpty())
        Result.push_back(Argument.Type().get());

    if (!ReturnType().isEmpty())
      Result.push_back(ReturnType().get());

    return Result;
  }
};

#include "revng/Model/Generated/Late/CABIFunctionDefinition.h"
