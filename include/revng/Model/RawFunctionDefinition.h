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
doc: |-
  The function type described by explicitly listing how arguments and return
  values are passed.

  This is a "low level" representation of the prototype of a function. Where
  the list of registers used to pass arguments and return values is explicitl.
  For stack arguments, they are collected in a single `struct`
  (`StackArgumentsType`).

  In contrast, a `CABIFunctionDefinition` expresses the function type from a
  high-level perspective (e.g., a single argument might span multiple registers)
  and his associated to a well-known ABI. Given an ABI, it is always possible
  to convert a `CABIFunctionDefinition` into a `RawFunctionDefinition`.
fields:
  - name: Architecture
    type: Architecture
    doc: The processor architecture of this function type.
  - name: Arguments
    optional: true
    sequence:
      type: SortedVector
      elementType: NamedTypedRegister
    doc: |-
      The list of registers used to pass arguments.

      The registers must belong to `Architecture`.
  - name: ReturnValues
    optional: true
    sequence:
      type: SortedVector
      elementType: NamedTypedRegister
    doc: |-
      The list of registers used to return values.

      The registers must belong to `Architecture`.
  - name: ReturnValueComment
    type: string
    optional: true
  - name: PreservedRegisters
    optional: true
    sequence:
      type: SortedVector
      elementType: Register
    doc: |-
      The list of registers preserved by functions using this function type.

      The registers must belong to `Architecture`.
  - name: FinalStackOffset
    type: uint64_t
    optional: true
    doc: |-
      The expected difference between the initial and final value of the stack
      pointer.

      For instance, in the x86-64 SystemV ABI, the difference between the
      initial and final value of the stack pointer is 8.
      This is due to the fact that `ret` instruction increase the stack pointer
      by 8.
  - name: StackArgumentsType
    doc: The type of the `struct` representing all of the stack arguments.
    type: Type
    optional: true
    upcastable: true
TUPLE-TREE-YAML */

#include "revng/Model/Generated/Early/RawFunctionDefinition.h"

class model::RawFunctionDefinition
  : public model::generated::RawFunctionDefinition {
public:
  using generated::RawFunctionDefinition::RawFunctionDefinition;

  NamedTypedRegister &addArgument(model::Register::Values Location,
                                  model::UpcastableType &&Type) {
    auto &&[Iterator, Success] = Arguments().emplace(Location);
    revng_assert(Success);
    Iterator->Type() = std::move(Type);
    return *Iterator;
  }
  NamedTypedRegister &addReturnValue(model::Register::Values Location,
                                     model::UpcastableType &&Type) {
    auto &&[Iterator, Success] = ReturnValues().emplace(Location);
    revng_assert(Success);
    Iterator->Type() = std::move(Type);
    return *Iterator;
  }

public:
  /// The helper for stack argument type unwrapping.
  /// Use this when you need to access/modify the existing struct,
  /// and \ref StackArgumentsType() when you need to assign a new one.
  model::StructDefinition *stackArgumentsType() {
    if (StackArgumentsType().isEmpty())
      return nullptr;
    else
      return &StackArgumentsType()->toStruct();
  }

  /// The helper for stack argument type unwrapping.
  /// Use this when you need to access/modify the existing struct,
  /// and \ref StackArgumentsType() when you need to assign a new one.
  const model::StructDefinition *stackArgumentsType() const {
    if (StackArgumentsType().isEmpty())
      return nullptr;
    else
      return &StackArgumentsType()->toStruct();
  }

public:
  const llvm::SmallVector<const model::Type *, 4> edges() const {
    llvm::SmallVector<const model::Type *, 4> Result;

    for (const auto &Argument : Arguments())
      if (!Argument.Type().isEmpty())
        Result.push_back(Argument.Type().get());

    for (const auto &ReturnValue : ReturnValues())
      if (!ReturnValue.Type().isEmpty())
        Result.push_back(ReturnValue.Type().get());

    if (!StackArgumentsType().isEmpty())
      Result.push_back(StackArgumentsType().get());

    return Result;
  }
  llvm::SmallVector<model::Type *, 4> edges() {
    llvm::SmallVector<model::Type *, 4> Result;

    for (auto &Argument : Arguments())
      if (!Argument.Type().isEmpty())
        Result.push_back(Argument.Type().get());

    for (auto &ReturnValue : ReturnValues())
      if (!ReturnValue.Type().isEmpty())
        Result.push_back(ReturnValue.Type().get());

    if (!StackArgumentsType().isEmpty())
      Result.push_back(StackArgumentsType().get());

    return Result;
  }
};

#include "revng/Model/Generated/Late/RawFunctionDefinition.h"
