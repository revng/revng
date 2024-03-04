/// \file DefaultPrototype.cpp

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/ABI/DefaultFunctionPrototype.h"
#include "revng/ABI/Definition.h"
#include "revng/ABI/FunctionType/Support.h"
#include "revng/Support/EnumSwitch.h"

static model::UpcastableType defaultPrototype(model::Binary &Binary,
                                              model::ABI::Values ABI) {
  auto [Definition, Type] = Binary.makeRawFunctionDefinition();

  const abi::Definition &Defined = abi::Definition::get(ABI);
  for (const auto &Register : Defined.GeneralPurposeArgumentRegisters()) {
    auto [Argument, Success] = Definition.Arguments().emplace(Register);
    revng_assert(Success);

    Argument->Type() = model::PrimitiveType::make(Register);
  }

  for (const auto &Register : Defined.GeneralPurposeReturnValueRegisters()) {
    auto [ReturnValue, Success] = Definition.ReturnValues().emplace(Register);
    revng_assert(Success);

    ReturnValue->Type() = model::PrimitiveType::make(Register);
  }

  for (const auto &Register : Defined.CalleeSavedRegisters())
    Definition.PreservedRegisters().insert(Register);

  Definition.FinalStackOffset() = getCallPushSize(Binary.Architecture());

  return Type;
}

using OptionalABI = std::optional<model::ABI::Values>;
model::UpcastableType
abi::registerDefaultFunctionPrototype(model::Binary &Binary,
                                      OptionalABI MaybeABI) {
  if (!MaybeABI.has_value())
    MaybeABI = Binary.DefaultABI();
  revng_assert(*MaybeABI != model::ABI::Invalid);
  return defaultPrototype(Binary, MaybeABI.value());
}
