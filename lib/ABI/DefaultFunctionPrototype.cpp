/// \file DefaultPrototype.cpp
/// \brief

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/ABI/DefaultFunctionPrototype.h"
#include "revng/ABI/Trait.h"
#include "revng/Support/EnumSwitch.h"

using namespace model;

constexpr static PrimitiveTypeKind::Values selectTypeKind(Register::Values) {
  // TODO: implement a way to determine the register type. At the very least
  // we should be able to differentiate GPRs from the vector registers.

  return PrimitiveTypeKind::PointerOrNumber;
}

static QualifiedType buildType(Register::Values Register) {
  PrimitiveTypeKind::Values Kind = selectTypeKind(Register);
  size_t Size = Register::getSize(Register);
  return QualifiedType::getPrimitiveType(Kind, Size);
}

template<ABI::Values ABI>
TypePath defaultPrototype(Binary &TheBinary) {
  UpcastableType NewType = makeType<RawFunctionType>();
  TypePath TypePath = TheBinary.recordNewType(std::move(NewType));
  auto &Prototype = *llvm::cast<RawFunctionType>(TypePath.get());

  for (const auto &Reg : abi::Trait<ABI>::GeneralPurposeArgumentRegisters) {
    NamedTypedRegister Argument(Reg);
    Argument.Type() = buildType(Reg);
    Prototype.Arguments().insert(Argument);
  }

  for (const auto &Register :
       abi::Trait<ABI>::GeneralPurposeReturnValueRegisters) {
    TypedRegister ReturnValue(Register);
    ReturnValue.Type() = buildType(Register);
    Prototype.ReturnValues().insert(ReturnValue);
  }

  for (const auto &Register : abi::Trait<ABI>::CalleeSavedRegisters)
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
  return skippingEnumSwitch<1>(*MaybeABI, [&]<ABI::Values A>() {
    return defaultPrototype<A>(Binary);
  });
}
