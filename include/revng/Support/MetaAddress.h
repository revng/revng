#ifndef METAADDRESS_H
#define METAADDRESS_H

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// LLVM includes
#include "llvm/ADT/Triple.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/Casting.h"

// Local libraries includes
#include "revng/Support/Debug.h"

namespace MetaAddressType {

enum Values : uint16_t { Invalid, Regular, ARMThumb };

inline const char *toString(Values V) {
  switch (V) {
  case Invalid:
    return "Invalid";
  case Regular:
    return "Regular";
  case ARMThumb:
    return "ARMThumb";
  }

  revng_abort();
}

} // namespace MetaAddressType

class MetaAddress {
private:
  uint64_t Address;
  uint32_t Epoch;
  uint16_t AddressSpace;
  MetaAddressType::Values Type;

public:
  explicit MetaAddress() { setInvalid(); }

private:
  explicit MetaAddress(uint64_t Address,
                       MetaAddressType::Values Type,
                       uint32_t Epoch,
                       uint16_t AddressSpace) :
    Address(Address), Epoch(Epoch), AddressSpace(AddressSpace), Type(Type) {
    revng_assert(verify());
  }

public:
  static MetaAddress invalid() { return MetaAddress(); }

public:
  static MetaAddress fromPC(llvm::Triple::ArchType Arch,
                            uint64_t PC,
                            uint32_t Epoch = 0,
                            uint16_t AddressSpace = 0) {
    MetaAddress Result(0, MetaAddressType::Regular, Epoch, AddressSpace);

    unsigned Alignment = 1;

    switch (Arch) {
    case llvm::Triple::arm:
      if ((PC & 1) == 1) {
        Result.Type = MetaAddressType::ARMThumb;
        // No need to check alignment, since we already know it's an odd number
        // and that's enough
      } else {
        Alignment = 4;
      }
      break;

    case llvm::Triple::aarch64:
    case llvm::Triple::mips:
    case llvm::Triple::mipsel:
      Alignment = 4;
      break;

    case llvm::Triple::systemz:
      Alignment = 2;
      break;

    default:
      break;
    }

    if (PC % Alignment != 0) {
      return MetaAddress::invalid();
    } else {
      return Result.replacePC(PC);
    }
  }

  static MetaAddress fromAbsolute(uint64_t Address,
                                  uint32_t Epoch = 0,
                                  uint16_t AddressSpace = 0) {
    return MetaAddress(Address, MetaAddressType::Regular, Epoch, AddressSpace);
  }

  static MetaAddress fromConstant(llvm::Value *V) {
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

    return Result;
  }

public:
  static llvm::StructType *getStruct(llvm::Module *M) {
    using namespace llvm;
    auto *InvalidAddress = M->getGlobalVariable("invalid_address", true);
    return cast<StructType>(InvalidAddress->getType()->getPointerElementType());
  }

  static llvm::GlobalVariable *createStructVariable(llvm::Module *M) {
    using namespace llvm;
    auto *MetaAddressStruct = createStruct(M->getContext());
    return new GlobalVariable(*M,
                              MetaAddressStruct,
                              false,
                              GlobalValue::InternalLinkage,
                              invalid().toConstant(MetaAddressStruct),
                              StringRef("invalid_address"));
  }

  llvm::Constant *toConstant(llvm::Type *Type) const {
    using namespace llvm;

    auto *Struct = cast<llvm::StructType>(Type);

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

public:
  bool operator==(const MetaAddress &Other) const {
    return tie() == Other.tie();
  }

  bool operator!=(const MetaAddress &Other) const {
    return not(*this == Other);
  }

  bool operator<(const MetaAddress &Other) const { return tie() < Other.tie(); }

  bool operator>(const MetaAddress &Other) const { return tie() > Other.tie(); }

  bool operator<=(const MetaAddress &Other) const {
    return tie() <= Other.tie();
  }

  bool operator>=(const MetaAddress &Other) const {
    return tie() >= Other.tie();
  }

  template<typename T>
  MetaAddress &operator+=(T Offset) {
    T NewAddress = Address + Offset;

    if ((Offset >= 0 and NewAddress < T(Address))
        or (Offset < 0 and NewAddress > T(Address))) {
      setInvalid();
    } else {
      Address = NewAddress;
      revng_assert(verify());
    }

    return *this;
  }
  template<typename T>
  MetaAddress &operator-=(T Offset) {
    T NewAddress = Address - Offset;

    if ((Offset < 0 and NewAddress < T(Address))
        or (Offset >= 0 and NewAddress > T(Address))) {
      setInvalid();
    } else {
      Address = NewAddress;
      revng_assert(verify());
    }

    return *this;
  }

  template<typename T>
  MetaAddress operator+(T Offset) const {
    MetaAddress Result = *this;
    Result += Offset;
    return Result;
  }

  template<typename T>
  MetaAddress operator-(T Offset) const {
    MetaAddress Result = *this;
    Result -= Offset;
    return Result;
  }

  uint64_t operator-(const MetaAddress &Other) const {
    revng_assert(this->AddressSpace == Other.AddressSpace);
    return Address - Other.Address;
  }

public:
  MetaAddress relocate(const MetaAddress &Other) const {
    revng_assert(Epoch == Other.Epoch and AddressSpace == Other.AddressSpace
                 and Type == Other.Type);
    MetaAddress Result = *this;
    Result.Address += Other.Address;
    return Result;
  }

  MetaAddress replacePC(uint64_t PC) const {
    MetaAddress Result = *this;
    Result.normalize(PC);
    revng_assert(Result.verify());
    return Result;
  }

  MetaAddress replaceAddress(uint64_t Address) const {
    MetaAddress Result = *this;
    Result.Address = Address;
    if (not Result.verify())
      return MetaAddress::invalid();
    else
      return Result;
  }

public:
  bool isInvalid() const { return Type == MetaAddressType::Invalid; }
  bool isValid() const { return not isInvalid(); }

  uint64_t address() const { return Address; }

  uint64_t asPC() const {
    revng_assert(Type != MetaAddressType::Invalid);
    return asPCOrZero();
  }

  uint64_t asPCOrZero() const {
    switch (Type) {
    case MetaAddressType::Invalid:
      return 0;
    case MetaAddressType::Regular:
      return Address;
    case MetaAddressType::ARMThumb:
      return Address | 1;
    }

    revng_abort();
  }

  bool isDefaultAddressSpace() const { return AddressSpace == 0; }
  uint16_t addressSpace() const { return AddressSpace; }

  bool isDefaultEpoch() const { return Epoch == 0; }
  uint32_t epoch() const { return Epoch; }

  MetaAddressType::Values type() const { return Type; }

public:
  bool verify() const debug_function {
    switch (Type) {
    case MetaAddressType::Invalid:
      return *this == invalid();
    case MetaAddressType::Regular:
      return true;
    case MetaAddressType::ARMThumb:
      return (Address & 1) == 0;
    }

    revng_abort();
  }

  void dump() const debug_function { dump(dbg); }

  template<typename T>
  void dump(T &Output) const {
    Output << std::hex << "0x" << asPC() << " ("
           << MetaAddressType::toString(Type) << " at 0x" << address() << ")";

    if (not isDefaultAddressSpace()) {
      Output << " in address space " << AddressSpace;
    }

    if (not isDefaultEpoch()) {
      Output << " at epoch " << Epoch;
    }

    revng_assert(verify());
  }

private:
  void normalize(uint64_t PC) {
    switch (Type) {
    case MetaAddressType::Invalid:
      revng_abort();

    case MetaAddressType::Regular:
      Address = PC;
      break;

    case MetaAddressType::ARMThumb:
      if ((PC & 1) == 0)
        setInvalid();
      else
        Address = PC & ~1;
      break;
    }
  }

  void setInvalid() {
    Type = MetaAddressType::Invalid;
    Address = 0;
    Epoch = 0;
    AddressSpace = 0;
  }

  static llvm::StructType *createStruct(llvm::LLVMContext &Context) {
    auto *Uint64Ty = llvm::Type::getInt64Ty(Context);
    auto *Uint32Ty = llvm::Type::getInt32Ty(Context);
    auto *Uint16Ty = llvm::Type::getInt16Ty(Context);
    return llvm::StructType::create({ Uint64Ty, Uint32Ty, Uint16Ty, Uint16Ty },
                                    "MetaAddress");
  }

private:
  using Tied = std::tuple<uint32_t, uint16_t, uint64_t, uint16_t>;
  Tied tie() const { return std::tie(Epoch, AddressSpace, Address, Type); }
};

static_assert(sizeof(MetaAddress) <= 128 / 8,
              "MetaAddress is larger than 128 bits");

#endif // METAADDRESS_H
