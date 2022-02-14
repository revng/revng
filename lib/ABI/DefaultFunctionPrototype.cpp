/// \file DefaultPrototype.cpp
/// \brief

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/ABI/DefaultFunctionPrototype.h"
#include "revng/ABI/Trait.h"
#include "revng/Support/EnumSwitch.h"

constexpr static model::PrimitiveTypeKind::Values
selectTypeKind(model::Register::Values) {
  // TODO: implement a way to determine the register type. At the very least
  // we should be able to differentiate GPRs from the vector registers.

  return model::PrimitiveTypeKind::PointerOrNumber;
}

static model::QualifiedType
buildType(model::Register::Values Register, model::Binary &TheBinary) {
  model::PrimitiveTypeKind::Values Kind = selectTypeKind(Register);
  size_t Size = model::Register::getSize(Register);
  return model::QualifiedType(TheBinary.getPrimitiveType(Kind, Size), {});
}

template<model::ABI::Values ABI>
model::TypePath defaultPrototype(model::Binary &TheBinary) {
  model::UpcastableType NewType = model::makeType<model::RawFunctionType>();
  model::TypePath TypePath = TheBinary.recordNewType(std::move(NewType));
  auto &Prototype = *llvm::cast<model::RawFunctionType>(TypePath.get());

  for (const auto &Reg : abi::Trait<ABI>::GeneralPurposeArgumentRegisters) {
    model::NamedTypedRegister Argument(Reg);
    Argument.Type = buildType(Reg, TheBinary);
    Prototype.Arguments.insert(Argument);
  }

  for (const auto &Rg : abi::Trait<ABI>::GeneralPurposeReturnValueRegisters) {
    model::TypedRegister ReturnValue(Rg);
    ReturnValue.Type = buildType(Rg, TheBinary);
    Prototype.ReturnValues.insert(ReturnValue);
  }

  for (const auto &Register : abi::Trait<ABI>::CalleeSavedRegisters)
    Prototype.PreservedRegisters.insert(Register);

  return TypePath;
}

model::TypePath
abi::defaultFunctionPrototype(model::Binary &BinaryToRecordTheTypeAt,
                              std::optional<model::ABI::Values> MaybeABI) {
  if (!MaybeABI.has_value())
    MaybeABI = BinaryToRecordTheTypeAt.DefaultABI;
  revng_assert(*MaybeABI != model::ABI::Invalid);
  return skippingEnumSwitch<1>(*MaybeABI, [&]<model::ABI::Values A>() {
    return defaultPrototype<A>(BinaryToRecordTheTypeAt);
  });
}
