/// \file MetaAddress.cpp
/// \brief Implementation of MetaAddress.

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// LLVM includes
#include "llvm/IR/Constants.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/raw_ostream.h"

// Local libraries includes
#include "revng/Support/MetaAddress.h"

using namespace llvm;

Constant *MetaAddress::toConstant(llvm::Type *Type) const {
  using namespace llvm;

  auto *Struct = cast<StructType>(Type);

  auto GetInt = [Struct](unsigned Index, uint64_t Value) {
    return ConstantInt::get(cast<IntegerType>(Struct->getElementType(Index)),
                            Value);
  };

  return ConstantStruct::get(Struct,
                             GetInt(0, this->Address),
                             GetInt(1, this->Epoch),
                             GetInt(2, this->AddressSpace),
                             GetInt(3, this->Type));
}

GlobalVariable *MetaAddress::createStructVariable(Module *M) {
  using namespace llvm;
  auto *MetaAddressStruct = createStruct(M->getContext());
  return new GlobalVariable(*M,
                            MetaAddressStruct,
                            false,
                            GlobalValue::InternalLinkage,
                            invalid().toConstant(MetaAddressStruct),
                            StringRef("invalid_address"));
}

StructType *MetaAddress::getStruct(Module *M) {
  using namespace llvm;
  auto *InvalidAddress = M->getGlobalVariable("invalid_address", true);
  return cast<StructType>(InvalidAddress->getType()->getPointerElementType());
}

MetaAddress MetaAddress::fromConstant(Value *V) {
  using namespace llvm;
  using namespace MetaAddressType;

  auto *Struct = cast<ConstantStruct>(V);
  revng_assert(Struct->getNumOperands() == 4);

  auto CI = [](Value *V) { return cast<ConstantInt>(V)->getLimitedValue(); };

  MetaAddress Result;
  Result.Address = CI(Struct->getOperand(0));
  Result.Epoch = CI(Struct->getOperand(1));
  Result.AddressSpace = CI(Struct->getOperand(2));
  Result.Type = static_cast<Values>(CI(Struct->getOperand(3)));
  Result.validate();

  return Result;
}

StructType *MetaAddress::createStruct(LLVMContext &Context) {
  auto *Uint64Ty = Type::getInt64Ty(Context);
  auto *Uint32Ty = Type::getInt32Ty(Context);
  auto *Uint16Ty = Type::getInt16Ty(Context);
  return StructType::create({ Uint64Ty, Uint32Ty, Uint16Ty, Uint16Ty },
                            "MetaAddress");
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
           << MetaAddressType::toString(Type);
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

  bool Success;
  uint64_t Value;

  Success = Parts[0].getAsInteger(0, Result.Address);
  if (not Success)
    return MetaAddress::invalid();

  Result.Type = MetaAddressType::fromString(Parts[1]);
  if (Result.Type == MetaAddressType::Invalid)
    return MetaAddress::invalid();

  Result.Epoch = 0;
  if (Parts.size() >= 3 and Parts[2].size() > 0) {
    Success = Parts[2].getAsInteger(0, Result.Epoch);
    if (not Success)
      return MetaAddress::invalid();
  }

  Result.AddressSpace = 0;
  if (Parts.size() == 4 and Parts[3].size() > 0) {
    Success = Parts[3].getAsInteger(0, Result.AddressSpace);
    if (not Success)
      return MetaAddress::invalid();
  }

  Result.validate();

  return Result;
}
