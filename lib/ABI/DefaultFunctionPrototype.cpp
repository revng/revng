/// \file DefaultPrototype.cpp

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/ABI/DefaultFunctionPrototype.h"
#include "revng/ABI/Definition.h"
#include "revng/Support/EnumSwitch.h"

using namespace model;

constexpr static PrimitiveTypeKind::Values selectTypeKind(Register::Values) {
  // TODO: implement a way to determine the register type. At the very least
  // we should be able to differentiate GPRs from the vector registers.

  return PrimitiveTypeKind::PointerOrNumber;
}

static QualifiedType buildType(Register::Values Register, Binary &TheBinary) {
  PrimitiveTypeKind::Values Kind = selectTypeKind(Register);
  size_t Size = Register::getSize(Register);
  return QualifiedType(TheBinary.getPrimitiveType(Kind, Size), {});
}

static TypePath defaultPrototype(Binary &TheBinary, model::ABI::Values ABI) {
  UpcastableType NewType = makeType<RawFunctionType>();
  TypePath TypePath = TheBinary.recordNewType(std::move(NewType));
  auto &Prototype = *llvm::cast<RawFunctionType>(TypePath.get());

  const abi::Definition &Defined = abi::Definition::get(ABI);
  for (const auto &Register : Defined.GeneralPurposeArgumentRegisters()) {
    NamedTypedRegister Argument(Register);
    Argument.Type() = buildType(Register, TheBinary);
    Prototype.Arguments().insert(Argument);
  }

  for (const auto &Register : Defined.GeneralPurposeReturnValueRegisters()) {
    TypedRegister ReturnValue(Register);
    ReturnValue.Type() = buildType(Register, TheBinary);
    Prototype.ReturnValues().insert(ReturnValue);
  }

  for (const auto &Register : Defined.CalleeSavedRegisters())
    Prototype.PreservedRegisters().insert(Register);

  using namespace Architecture;
  Prototype.FinalStackOffset() = getCallPushSize(TheBinary.Architecture());

  return TypePath;
}

model::TypePath
abi::registerDefaultFunctionPrototype(Binary &Binary,
                                      std::optional<ABI::Values> MaybeABI) {
  if (!MaybeABI.has_value())
    MaybeABI = Binary.DefaultABI();
  revng_assert(*MaybeABI != ABI::Invalid);
  return defaultPrototype(Binary, MaybeABI.value());
}
