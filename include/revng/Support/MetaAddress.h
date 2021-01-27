#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/ADT/Triple.h"

#include "revng/Support/Debug.h"

namespace llvm {
class Type;
class Constant;
class ConstantInt;
class Value;
class LLVMContext;
class Module;
class StructType;
class GlobalVariable;
class Instruction;
class ConstantFolder;
class IRBuilderDefaultInserter;

template<typename T /* = ConstantFolder */,
         typename Inserter /* = IRBuilderDefaultInserter */>
class IRBuilder;
} // namespace llvm

namespace MetaAddressType {

enum Values : uint16_t {
  /// An invalid address
  Invalid,

  /// A 32-bit generic address
  Generic32,

  /// A 64-bit generic address
  Generic64,

  /// The address of a x86 basic block
  Code_x86,

  /// The address of a x86-64 basic block
  Code_x86_64,

  /// The address of a MIPS basic block
  Code_mips,

  /// The address of a MIPS little-endian basic block
  Code_mipsel,

  /// The address of a regular ARM basic block
  Code_arm,

  /// The address of a ARM Thumb basic block
  Code_arm_thumb,

  /// The address of a AArch64 basic block
  Code_aarch64,

  /// The address of a z/Architecture (s390x) basic block
  Code_systemz

};

inline bool isValid(Values V) {
  switch (V) {
  case Invalid:
  case Generic32:
  case Generic64:
  case Code_x86:
  case Code_x86_64:
  case Code_mips:
  case Code_mipsel:
  case Code_arm:
  case Code_arm_thumb:
  case Code_aarch64:
  case Code_systemz:
    return true;
  default:
    return false;
  }
}

inline const char *toString(Values V) {
  switch (V) {
  case Invalid:
    return "Invalid";
  case Generic32:
    return "Generic32";
  case Generic64:
    return "Generic64";
  case Code_x86:
    return "Code_x86";
  case Code_x86_64:
    return "Code_x86_64";
  case Code_mips:
    return "Code_mips";
  case Code_mipsel:
    return "Code_mipsel";
  case Code_arm:
    return "Code_arm";
  case Code_arm_thumb:
    return "Code_arm_thumb";
  case Code_aarch64:
    return "Code_aarch64";
  case Code_systemz:
    return "Code_systemz";
  }

  revng_abort();
}

inline Values fromString(llvm::StringRef String) {
  if (String == "Generic32") {
    return Generic32;
  } else if (String == "Generic64") {
    return Generic64;
  } else if (String == "Code_x86") {
    return Code_x86;
  } else if (String == "Code_x86_64") {
    return Code_x86_64;
  } else if (String == "Code_mips") {
    return Code_mips;
  } else if (String == "Code_mipsel") {
    return Code_mipsel;
  } else if (String == "Code_arm") {
    return Code_arm;
  } else if (String == "Code_arm_thumb") {
    return Code_arm_thumb;
  } else if (String == "Code_aarch64") {
    return Code_aarch64;
  } else if (String == "Code_systemz") {
    return Code_systemz;
  } else {
    return Invalid;
  }

  revng_abort();
}

inline const llvm::Optional<llvm::Triple::ArchType> arch(Values V) {
  switch (V) {
  case Code_x86:
    return llvm::Triple::x86;
  case Code_x86_64:
    return llvm::Triple::x86_64;
  case Code_mips:
    return llvm::Triple::mips;
  case Code_mipsel:
    return llvm::Triple::mipsel;
  case Code_arm:
  case Code_arm_thumb:
    return llvm::Triple::arm;
  case Code_aarch64:
    return llvm::Triple::aarch64;
  case Code_systemz:
    return llvm::Triple::systemz;
  case Invalid:
  case Generic32:
  case Generic64:
    return {};
  default:
    revng_abort();
  }
}

/// Returns Generic32 or Generic64 depending on the size of addresses in \p Arch
inline Values genericFromArch(llvm::Triple::ArchType Arch) {
  switch (Arch) {
  case llvm::Triple::x86:
  case llvm::Triple::arm:
  case llvm::Triple::mips:
  case llvm::Triple::mipsel:
    return Generic32;
  case llvm::Triple::x86_64:
  case llvm::Triple::aarch64:
  case llvm::Triple::systemz:
    return Generic64;
  default:
    revng_abort("Unsupported architecture");
  }

  revng_abort("Unsupported architecture");
}

/// Convert \p Type to the corresponding generic type
inline Values toGeneric(Values Type) {
  switch (Type) {
  case Invalid:
    revng_abort("Can't convert to generic an invalid type");

  case Generic32:
  case Generic64:
    return Type;

  case Code_x86:
  case Code_arm_thumb:
  case Code_mips:
  case Code_mipsel:
  case Code_arm:
    return Generic32;

  case Code_x86_64:
  case Code_systemz:
  case Code_aarch64:
    return Generic64;
  }

  revng_abort("Unsupported architecture");
}

/// Get the default type for code of the given architecture
inline Values defaultCodeFromArch(llvm::Triple::ArchType Arch) {
  switch (Arch) {
  case llvm::Triple::x86:
    return Code_x86;
  case llvm::Triple::arm:
    return Code_arm;
  case llvm::Triple::mips:
    return Code_mips;
  case llvm::Triple::mipsel:
    return Code_mipsel;
  case llvm::Triple::x86_64:
    return Code_x86_64;
  case llvm::Triple::aarch64:
    return Code_aarch64;
  case llvm::Triple::systemz:
    return Code_systemz;
  default:
    revng_abort("Unsupported architecture");
  }
}

/// Get the alignment of the corresponding type
///
/// \note Generic types have alignment of 1
inline unsigned alignment(Values Type) {
  switch (Type) {
  case Invalid:
    revng_abort("Invalid addresses have no alignment");
  case Generic32:
  case Generic64:
  case Code_x86:
  case Code_x86_64:
    return 1;
  case Code_arm_thumb:
  case Code_systemz:
    return 2;
  case Code_mips:
  case Code_mipsel:
  case Code_arm:
  case Code_aarch64:
    return 4;
  }

  revng_abort();
}

/// Get the size in bit of an address of the given type
inline unsigned bitSize(Values Type) {
  switch (Type) {
  case Invalid:
    revng_abort("Invalid addresses have no bit size");
  case Generic32:
  case Code_x86:
  case Code_arm_thumb:
  case Code_mips:
  case Code_mipsel:
  case Code_arm:
    return 32;
  case Generic64:
  case Code_x86_64:
  case Code_systemz:
  case Code_aarch64:
    return 64;
  }

  revng_abort();
}

/// Get a 64-bits mask representing the relevant bits for the given type
///
/// \note The alignment is not considered in this mask.
inline uint64_t addressMask(Values Type) {
  revng_assert(bitSize(Type) != 0);
  return std::numeric_limits<uint64_t>::max() >> (64 - bitSize(Type));
}

/// Does \p Type represent a code address?
inline bool isCode(Values Type) {
  switch (Type) {
  case Invalid:
  case Generic32:
  case Generic64:
    return false;

  case Code_x86:
  case Code_arm_thumb:
  case Code_mips:
  case Code_mipsel:
  case Code_arm:
  case Code_x86_64:
  case Code_systemz:
  case Code_aarch64:
    return true;
  }

  revng_abort();
}

/// Does \p Type represent an address pointing to \p Arch code?
inline bool isCode(Values Type, llvm::Triple::ArchType Arch) {
  switch (Arch) {
  case llvm::Triple::x86:
    return Type == Code_x86;
  case llvm::Triple::arm:
    return Type == Code_arm or Type == Code_arm_thumb;
  case llvm::Triple::mips:
    return Type == Code_mips;
  case llvm::Triple::mipsel:
    return Type == Code_mipsel;
  case llvm::Triple::x86_64:
    return Type == Code_x86_64;
  case llvm::Triple::aarch64:
    return Type == Code_aarch64;
  case llvm::Triple::systemz:
    return Type == Code_systemz;
  default:
    revng_abort("Unsupported architecture");
  }

  revng_abort();
}

/// Is \p Type a generic address?
inline bool isGeneric(Values Type) {
  switch (Type) {
  case Invalid:
  case Code_x86:
  case Code_arm_thumb:
  case Code_mips:
  case Code_mipsel:
  case Code_arm:
  case Code_x86_64:
  case Code_systemz:
  case Code_aarch64:
    return false;

  case Generic32:
  case Generic64:
    return true;
  }

  revng_abort();
}

inline bool isDefaultCode(Values Type) {
  switch (Type) {
  case Code_x86:
  case Code_mips:
  case Code_mipsel:
  case Code_arm:
  case Code_x86_64:
  case Code_systemz:
  case Code_aarch64:
    return true;

  case Invalid:
  case Generic32:
  case Generic64:
  case Code_arm_thumb:
    return false;
  }

  revng_abort();
}

} // namespace MetaAddressType

/// Represents an address with a type, an address space and epoch
///
/// MetaAddress is a uint64_t on steroids.
///
/// Its key goal is to allow to distinguish different things at the same address
/// (e.g., regular and Thumb code at the same address).  It also provides
/// appropriate arithmetic depending on the address type.
///
/// MetaAddress represents four things:
///
/// 1. The absolute value of the address
/// 2. The "epoch": a progressive identifier that represents a timestamp. It
///    enables users to represent the fact that at the same address there might
///    be different things at different points in time. Its main purpose is to
///    represent self-modifying code.
/// 3. The address space: a generic identifier for architectures that have
///    access to multiple address spaces.
/// 4. The type: a MetaAddress can be used to represent a generic address or to
///    code. See MetaAddressType for further details.
///
/// \note Generic addresses have no alignment constraints.
class MetaAddress {
  friend class ProgramCounterHandler;

private:
  uint32_t Epoch;
  uint16_t AddressSpace;
  MetaAddressType::Values Type;
  uint64_t Address;

public:
  /// \name Constructors
  ///
  /// @{

  /// Public constructor creating an invalid MetaAddress
  ///
  /// \note Prefer MetaAddress::invalid()
  explicit MetaAddress() :
    Epoch(0), AddressSpace(0), Type(MetaAddressType::Invalid), Address(0) {}

  /// Public constructor allowing to create a custom instance to validate
  ///
  /// \note Prefer MetaAddress:fromPC or MetaAddress::fromGeneric
  explicit MetaAddress(uint64_t Address,
                       MetaAddressType::Values Type,
                       uint32_t Epoch = 0,
                       uint16_t AddressSpace = 0) :
    Epoch(Epoch), AddressSpace(AddressSpace), Type(Type), Address(Address) {

    // Verify the given data
    validate();
  }

  /// @}

public:
  /// \name Factory methods
  ///
  /// @{

  /// Create an invalid MetaAddress
  static MetaAddress invalid() { return MetaAddress(); }

  /// Create a MetaAddress from a pointer to \p Arch code
  static MetaAddress fromPC(llvm::Triple::ArchType Arch,
                            uint64_t PC,
                            uint32_t Epoch = 0,
                            uint16_t AddressSpace = 0) {

    // Create the base MetaAddress, point at code at zero
    MetaAddress Result(0,
                       MetaAddressType::defaultCodeFromArch(Arch),
                       Epoch,
                       AddressSpace);

    // A code MetaAddress pointing at 0 should always be valid
    revng_assert(Result.isValid());

    if (Arch == llvm::Triple::arm and (PC & 1) == 1) {
      // A pointer to ARM code with the LSB turned on is Thumb code

      // Override the type
      Result.Type = MetaAddressType::Code_arm_thumb;
    }

    Result.setPC(PC);

    // Check alignment
    Result.validate();

    return Result;
  }

  static MetaAddress fromPC(MetaAddress Base, uint64_t Address) {
    return fromPC(*Base.arch(), Address, Base.epoch(), Base.addressSpace());
  }

  /// Create a generic MetaAddress for architecture \p Arch
  static MetaAddress fromGeneric(llvm::Triple::ArchType Arch,
                                 uint64_t Address,
                                 uint32_t Epoch = 0,
                                 uint16_t AddressSpace = 0) {
    return MetaAddress(Address,
                       MetaAddressType::genericFromArch(Arch),
                       Epoch,
                       AddressSpace);
  }

  /// @}

public:
  /// \name llvm::ConstantStruct (de-)serialization methods
  ///
  /// @{

  /// Create an llvm::StructType to be used with fromConstant and toConstant
  static llvm::StructType *createStruct(llvm::LLVMContext &Context);

  /// Create a global variable named "invalid_address" of type createStruct
  static llvm::GlobalVariable *createStructVariable(llvm::Module *M);

  /// Get the type of the "invalid_address" global variable
  static llvm::StructType *getStruct(llvm::Module *M);

  /// Deserialize a MetaAddress from an llvm::ConstantStruct
  static MetaAddress fromConstant(llvm::Value *V);

  /// Serialize a MetaAddress to an llvm::StructType
  llvm::Constant *toConstant(llvm::Type *Type) const;

  /// @}

public:
  using IRBuilderType = llvm::IRBuilder<llvm::ConstantFolder,
                                        llvm::IRBuilderDefaultInserter>;

  static llvm::Instruction *composeIntegerPC(IRBuilderType &B,
                                             llvm::Value *AddressValue,
                                             llvm::Value *EpochValue,
                                             llvm::Value *AddressSpaceValue,
                                             llvm::Value *TypeValue);

  static MetaAddress decomposeIntegerPC(llvm::ConstantInt *Value);

public:
  /// If isCode(), let this decay to the corresponding generic address
  MetaAddress toGeneric() const {
    revng_check(isValid());

    MetaAddress Result = *this;
    Result.Type = MetaAddressType::toGeneric(Type);
    return Result;
  }

  MetaAddress toPC(llvm::Triple::ArchType Arch) const {
    return fromPC(Arch, Address, Epoch, AddressSpace);
  }

public:
  /// @{
  bool operator==(const MetaAddress &Other) const {
    return tie() == Other.tie();
  }

  bool operator!=(const MetaAddress &Other) const {
    return not(*this == Other);
  }

  bool operator<(const MetaAddress &Other) const { return tie() < Other.tie(); }
  bool operator<=(const MetaAddress &Other) const {
    return tie() <= Other.tie();
  }
  bool operator>(const MetaAddress &Other) const { return tie() > Other.tie(); }
  bool operator>=(const MetaAddress &Other) const {
    return tie() >= Other.tie();
  }

  /// @}

  /// \name Address comparisons
  ///
  /// Comparison operators are defined only if
  /// this->addressIsComparableWith(Other)
  ///
  /// @{

  /// Is this address comparable with \p Other
  ///
  /// Two MetaAddresses are comparable if they are both valid, they refer to the
  /// same address space and they have the same size in bits.
  bool addressIsComparableWith(const MetaAddress &Other) const {
    return (isValid() and Other.isValid() and AddressSpace == Other.AddressSpace
            and bitSize() == Other.bitSize());
  }

  bool addressEquals(const MetaAddress &Other) const {
    revng_check(addressIsComparableWith(Other));
    return Address == Other.Address;
  }

  bool addressDiffers(const MetaAddress &Other) const {
    return !addressEquals(Other);
  }

  bool addressLowerThan(const MetaAddress &Other) const {
    revng_check(addressIsComparableWith(Other));
    return Address < Other.Address;
  }

  bool addressLowerThanOrEqual(const MetaAddress &Other) const {
    revng_check(addressIsComparableWith(Other));
    return Address <= Other.Address;
  }

  bool addressGreaterThanOrEqual(const MetaAddress &Other) const {
    return not(addressLowerThan(Other));
  }

  bool addressGreaterThan(const MetaAddress &Other) const {
    return not(addressLowerThanOrEqual(Other));
  }

  uint64_t operator-(const MetaAddress &Other) const {
    revng_check(addressIsComparableWith(Other));
    return Address - Other.Address;
  }
  /// @}

  /// \name Arithmetic additions/subtractions
  ///
  /// @{
  MetaAddress &operator+=(uint64_t Offset) {
    revng_check(isValid());
    setAddress(Address + Offset);
    return *this;
  }

  MetaAddress &operator-=(uint64_t Offset) {
    revng_check(isValid());
    setAddress(Address - Offset);
    return *this;
  }

  MetaAddress operator+(uint64_t Offset) const {
    MetaAddress Result = *this;
    Result += Offset;
    return Result;
  }

  MetaAddress operator-(uint64_t Offset) const {
    MetaAddress Result = *this;
    Result -= Offset;
    return Result;
  }
  /// @}

public:
  /// Build a new MetaAddress replacing the address with a new (valid) address
  ///
  /// The given address must be valid for the current type. The resulting type
  /// has the same epoch, type and address space as this.
  MetaAddress replaceAddress(uint64_t Address) const {
    revng_check(isValid());

    MetaAddress Result = *this;
    Result.setAddress(Address);
    Result.validate();
    return Result;
  }

public:
  /// \name Accessors
  ///
  /// @{
  uint64_t address() const {
    revng_assert(isValid());
    return Address;
  }

  /// Return the wrapped address in its PC representation
  ///
  /// \note Don't call this method if `!(isValid() && isCode())`
  uint64_t asPC() const {
    revng_check(isValid());
    return asPCOrZero();
  }

  /// Return the wrapped address in its PC representation, or 0 if invalid
  uint64_t asPCOrZero() const {
    revng_check(isCode() or isInvalid());

    switch (Type) {
    case MetaAddressType::Invalid:
      revng_assert(Address == 0);
      return 0;

    case MetaAddressType::Code_arm_thumb:
      revng_assert((Address & 1) == 0);
      return Address | 1;

    case MetaAddressType::Code_x86:
    case MetaAddressType::Code_x86_64:
    case MetaAddressType::Code_systemz:
    case MetaAddressType::Code_mips:
    case MetaAddressType::Code_mipsel:
    case MetaAddressType::Code_arm:
    case MetaAddressType::Code_aarch64:
      return Address;

    case MetaAddressType::Generic32:
    case MetaAddressType::Generic64:
      revng_abort();
    }

    revng_abort();
  }

  uint16_t addressSpace() const {
    revng_check(isValid());
    return AddressSpace;
  }
  bool isDefaultAddressSpace() const { return addressSpace() == 0; }

  uint32_t epoch() const {
    revng_check(isValid());
    return Epoch;
  }
  bool isDefaultEpoch() const { return epoch() == 0; }

  MetaAddressType::Values type() const { return Type; }
  bool isInvalid() const { return Type == MetaAddressType::Invalid; }
  bool isValid() const { return not isInvalid(); }
  bool isCode() const { return MetaAddressType::isCode(Type); }
  bool isCode(llvm::Triple::ArchType Arch) const {
    return MetaAddressType::isCode(Type, Arch);
  }
  bool isGeneric() const { return MetaAddressType::isGeneric(Type); }
  unsigned bitSize() const { return MetaAddressType::bitSize(Type); }
  unsigned alignment() const { return MetaAddressType::alignment(Type); }

  llvm::Optional<llvm::Triple::ArchType> arch() {
    return MetaAddressType::arch(Type);
  }

  bool isDefaultCode() const { return MetaAddressType::isDefaultCode(Type); }

  /// @}

public:
  void dump() const debug_function { dump(dbg); }

  template<typename T>
  void dump(T &Output) const {
    dumpInternal(Output, Address);
  }

  template<typename T>
  void dumpRelativeTo(T &Output,
                      const MetaAddress &Base,
                      llvm::StringRef BaseName) const {
    Output << BaseName.data();

    if (Base == *this)
      return;

    Output << ".";

    dumpInternal(Output, Address - Base.Address);
  }

public:
  MetaAddress pageStart() const {
    revng_check(isValid());
    return toGeneric() - (Address % 4096);
  }

  MetaAddress nextPageStart() const {
    revng_check(isValid());
    return toGeneric() + (((Address + (4096 - 1)) / 4096) * 4096 - Address);
  }

private:
  bool verify() const debug_function {
    // Invalid addresses are all the same
    if (Type == MetaAddressType::Invalid) {
      return *this == invalid();
    }

    // Check alignment
    if (Address % alignment() != 0)
      return false;

    // Check address mask
    if (Address != (Address & addressMask()))
      return false;

    return true;
  }

  void validate() {
    if (not verify())
      setInvalid();
  }

  void setInvalid() { *this = MetaAddress(); }

  uint64_t addressMask() const { return MetaAddressType::addressMask(Type); }

  void setPC(uint64_t PC) {
    if (Type == MetaAddressType::Code_arm_thumb) {

      if ((PC & 1) == 0) {
        setInvalid();
        return;
      }

      PC = PC & ~1;
    }

    setAddress(PC);
  }

  void setAddress(uint64_t NewAddress) {
    Address = NewAddress & addressMask();
    validate();
  }

private:
  template<typename T>
  void dumpInternal(T &Output, uint64_t EffectiveAddress) const {
    Output << std::hex << "0x" << EffectiveAddress;

    if (not isDefaultAddressSpace()) {
      Output << "_as" << AddressSpace;
    }

    if (not isDefaultEpoch()) {
      Output << "_epoch" << Epoch;
    }

    if (not isDefaultCode()) {
      Output << "_" << MetaAddressType::toString(Type);
    }
  }

public:
  std::string toString() const;
  static MetaAddress fromString(llvm::StringRef Text);

private:
  using Tied = std::tuple<uint32_t, uint16_t, uint16_t, uint64_t>;
  Tied tie() const { return std::tie(Epoch, AddressSpace, Type, Address); }
};

static_assert(sizeof(MetaAddress) <= 128 / 8,
              "MetaAddress is larger than 128 bits");

template<typename T>
struct compareAddress {};

template<>
struct compareAddress<MetaAddress> {
  bool operator()(const MetaAddress &LHS, const MetaAddress &RHS) const {
    return LHS.addressLowerThan(RHS);
  }
};
