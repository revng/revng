/// \file DefaultPrototype.cpp

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/ABI/DefaultFunctionPrototype.h"
#include "revng/ABI/Definition.h"
#include "revng/Support/EnumSwitch.h"

using namespace model;

constexpr static PrimitiveKind::Values selectTypeKind(Register::Values) {
  // TODO: implement a way to determine the register type. At the very least
  // we should be able to differentiate GPRs from the vector registers.

  return PrimitiveKind::PointerOrNumber;
}

static QualifiedType buildType(Register::Values Register, Binary &TheBinary) {
  PrimitiveKind::Values Kind = selectTypeKind(Register);
  uint64_t Size = Register::getSize(Register);
  return QualifiedType(TheBinary.getPrimitiveType(Kind, Size), {});
}

static TypeDefinitionPath defaultPrototype(Binary &TheBinary,
                                           model::ABI::Values ABI) {
  auto [Prototype,
        Path] = TheBinary.makeTypeDefinition<RawFunctionDefinition>();

  revng_assert(ABI != model::ABI::Invalid);
  Prototype.Architecture() = model::ABI::getArchitecture(ABI);

  const abi::Definition &Defined = abi::Definition::get(ABI);
  for (const auto &Register : Defined.GeneralPurposeArgumentRegisters()) {
    NamedTypedRegister Argument(Register);
    Argument.Type() = buildType(Register, TheBinary);
    Prototype.Arguments().insert(Argument);
  }

  for (const auto &Register : Defined.GeneralPurposeReturnValueRegisters()) {
    NamedTypedRegister ReturnValue(Register);
    ReturnValue.Type() = buildType(Register, TheBinary);
    Prototype.ReturnValues().insert(ReturnValue);
  }

  for (const auto &Register : Defined.CalleeSavedRegisters())
    Prototype.PreservedRegisters().insert(Register);

  using namespace Architecture;
  Prototype.FinalStackOffset() = getCallPushSize(TheBinary.Architecture());

  return Path;
}

model::TypeDefinitionPath
abi::registerDefaultFunctionPrototype(Binary &Binary,
                                      std::optional<ABI::Values> MaybeABI) {
  if (!MaybeABI.has_value())
    MaybeABI = Binary.DefaultABI();
  revng_assert(*MaybeABI != ABI::Invalid);
  return defaultPrototype(Binary, MaybeABI.value());
}
