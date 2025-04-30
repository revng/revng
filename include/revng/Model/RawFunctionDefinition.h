#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/Model/NamedTypedRegister.h"
#include "revng/Model/TypeDefinition.h"

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
