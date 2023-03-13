/// \file MetaAddress.cpp
/// \brief Implementation of MetaAddress.

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/IR/Constants.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/raw_ostream.h"

#include "revng/Support/IRHelpers.h"
#include "revng/Support/MetaAddress.h"

using namespace llvm;

Constant *MetaAddress::toValue(llvm::Module *M) const {
  return getUniqueString(M, toString());
}

MetaAddress MetaAddress::fromValue(Value *V) {
  return MetaAddress::fromString(extractFromConstantStringPtr(V));
}

Instruction *MetaAddress::composeIntegerPC(IRBuilder<> &B,
                                           Value *AddressValue,
                                           Value *EpochValue,
                                           Value *AddressSpaceValue,
                                           Value *TypeValue) {
  llvm::Type *Int128 = B.getInt128Ty();
  auto Load128 = [&B, Int128](Value *V) { return B.CreateZExt(V, Int128); };

  auto *Epoch = B.CreateShl(Load128(EpochValue), 64);
  auto *AddressSpace = B.CreateShl(Load128(AddressSpaceValue), 64 + 32);
  auto *Type = B.CreateShl(Load128(TypeValue), 64 + 32 + 16);
  auto *Composed = B.CreateAdd(Load128(AddressValue),
                               B.CreateAdd(Epoch,
                                           B.CreateAdd(AddressSpace, Type)));

  return cast<Instruction>(Composed);
}

MetaAddress MetaAddress::decomposeIntegerPC(ConstantInt *Value) {
  revng_assert(Value->getType()->getBitWidth() == 128);
  const APInt &APValue = Value->getValue();
  uint64_t Lower = (APValue & UINT64_MAX).getLimitedValue();
  uint64_t Upper = APValue.lshr(64).getLimitedValue();

  MetaAddress Result;
  Result.Address = Lower;
  Result.Epoch = Upper & 0xFFFFFFFF;
  Result.AddressSpace = (Upper >> 32) & 0xFFFF;
  Result.Type = static_cast<MetaAddressType::Values>(Upper >> (32 + 16));
  Result.validate();

  return Result;
}

#define SEP ":"

std::string MetaAddress::toString() const {
  if (isInvalid())
    return SEP "Invalid";

  std::string Result;
  {
    raw_string_ostream Stream(Result);
    Stream << "0x" << Twine::utohexstr(Address) << SEP
           << MetaAddressType::toString(type());
    if (not isDefaultEpoch())
      Stream << SEP << Epoch;
    if (not isDefaultAddressSpace())
      Stream << SEP << AddressSpace;
  }

  return Result;
}

MetaAddress MetaAddress::fromString(StringRef Text) {
  if (Text == SEP "Invalid")
    return MetaAddress::invalid();

  MetaAddress Result;

  SmallVector<StringRef, 4> Parts;
  Text.split(Parts, SEP);

  if (Parts.size() < 2 or Parts.size() > 4)
    return MetaAddress::invalid();

  bool Error;
  uint64_t Value;

  Error = Parts[0].getAsInteger(0, Result.Address);
  if (Error)
    return MetaAddress::invalid();

  Result.Type = MetaAddressType::fromString(Parts[1]);
  if (Result.type() == MetaAddressType::Invalid)
    return MetaAddress::invalid();

  Result.Epoch = 0;
  if (Parts.size() >= 3 and Parts[2].size() > 0) {
    Error = Parts[2].getAsInteger(0, Result.Epoch);
    if (Error)
      return MetaAddress::invalid();
  }

  Result.AddressSpace = 0;
  if (Parts.size() == 4 and Parts[3].size() > 0) {
    Error = Parts[3].getAsInteger(0, Result.AddressSpace);
    if (Error)
      return MetaAddress::invalid();
  }

  Result.validate();

  return Result;
}
