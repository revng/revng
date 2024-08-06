#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Triple.h"

#include "revng/ADT/KeyedObjectContainer.h"
#include "revng/Support/Debug.h"
#include "revng/Support/OverflowSafeInt.h"

extern "C" {
#include "revng/Runtime/PlainMetaAddress.h"
}

namespace llvm {
class TypeDefinition;
class Constant;
class ConstantInt;
class Value;
class LLVMContext;
class Module;
class StructType;
class GlobalVariable;
class Instruction;
class IRBuilderBase;
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
  Code_systemz,

  /// The address of a hexagon basic block
  Code_hexagon
};

inline constexpr bool isValid(Values V) {
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
  case Code_hexagon:
    return true;
  default:
    return false;
  }
}

inline constexpr const char *toString(Values V) {
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
  case Code_hexagon:
    return "Code_hexagon";
  }

  revng_abort();
}

inline constexpr Values fromString(llvm::StringRef String) {
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
  } else if (String == "Code_hexagon") {
    return Code_hexagon;
  } else {
    return Invalid;
  }

  revng_abort();
}

inline constexpr const std::optional<llvm::Triple::ArchType> arch(Values V) {
  switch (V) {
  case Code_x86:
    return { llvm::Triple::x86 };
  case Code_x86_64:
    return { llvm::Triple::x86_64 };
  case Code_mips:
    return { llvm::Triple::mips };
  case Code_mipsel:
    return { llvm::Triple::mipsel };
  case Code_arm:
  case Code_arm_thumb:
    return { llvm::Triple::arm };
  case Code_aarch64:
    return { llvm::Triple::aarch64 };
  case Code_systemz:
    return { llvm::Triple::systemz };
  case Code_hexagon:
    return { llvm::Triple::hexagon };
  case Invalid:
  case Generic32:
  case Generic64:
    return {};
  default:
    revng_abort();
  }
}

/// Returns Generic32 or Generic64 depending on the size of addresses in \p Arch
inline constexpr Values genericFromArch(llvm::Triple::ArchType Arch) {
  switch (Arch) {
  case llvm::Triple::x86:
  case llvm::Triple::arm:
  case llvm::Triple::mips:
  case llvm::Triple::mipsel:
  case llvm::Triple::hexagon:
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
inline constexpr Values toGeneric(Values Type) {
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
  case Code_hexagon:
    return Generic32;

  case Code_x86_64:
  case Code_systemz:
  case Code_aarch64:
    return Generic64;
  }

  revng_abort("Unsupported architecture");
}

/// Get the default type for code of the given architecture
inline constexpr Values defaultCodeFromArch(llvm::Triple::ArchType Arch) {
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
  case llvm::Triple::hexagon:
    return Code_hexagon;
  default:
    revng_abort("Unsupported architecture");
  }
}

/// Get the alignment of the corresponding type
///
/// \note Generic types have alignment of 1
inline constexpr unsigned alignment(Values Type) {
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
  case Code_hexagon: // TODO(anjo):
    return 4;
  }

  revng_abort();
}

/// Get the size in bit of an address of the given type
inline constexpr unsigned bitSize(Values Type) {
  switch (Type) {
  case Invalid:
    revng_abort("Invalid addresses have no bit size");
  case Generic32:
  case Code_x86:
  case Code_arm_thumb:
  case Code_mips:
  case Code_mipsel:
  case Code_arm:
  case Code_hexagon:
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
inline constexpr uint64_t addressMask(Values Type) {
  revng_assert(bitSize(Type) != 0);
  return std::numeric_limits<uint64_t>::max() >> (64 - bitSize(Type));
}

/// Does \p Type represent a code address?
inline constexpr bool isCode(Values Type) {
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
  case Code_hexagon:
    return true;
  }

  revng_abort();
}

/// Does \p Type represent an address pointing to \p Arch code?
inline constexpr bool isCode(Values Type, llvm::Triple::ArchType Arch) {
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
  case llvm::Triple::hexagon:
    return Type == Code_hexagon;
  default:
    revng_abort("Unsupported architecture");
  }

  revng_abort();
}

/// Is \p Type a generic address?
inline constexpr bool isGeneric(Values Type) {
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
  case Code_hexagon:
    return false;

  case Generic32:
  case Generic64:
    return true;
  }

  revng_abort();
}

inline constexpr bool isDefaultCode(Values Type) {
  switch (Type) {
  case Code_x86:
  case Code_mips:
  case Code_mipsel:
  case Code_arm:
  case Code_x86_64:
  case Code_systemz:
  case Code_aarch64:
  case Code_hexagon:
    return true;

  case Invalid:
  case Generic32:
  case Generic64:
  case Code_arm_thumb:
    return false;
  }

  revng_abort();
}

inline constexpr llvm::StringRef getLLVMCPUFeatures(Values Type) {
  switch (Type) {
  case Code_arm_thumb:
    return "+thumb-mode";
  case Invalid:
  case Generic32:
  case Generic64:
  case Code_x86:
  case Code_mips:
  case Code_mipsel:
  case Code_arm:
  case Code_x86_64:
  case Code_systemz:
  case Code_aarch64:
  case Code_hexagon:
    return "";
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
class MetaAddress : private PlainMetaAddress {
private:
  friend class ProgramCounterHandler;

public:
  constexpr static llvm::StringRef Separator = ":";

public:
  class Features {
  public:
    llvm::Triple::ArchType Architecture = llvm::Triple::UnknownArch;
    uint32_t Epoch = 0;
    uint16_t AddressSpace = 0;

  public:
    bool operator==(const Features &Other) const = default;
  };

public:
  /// \name Constructors
  ///
  /// @{

  /// Public constructor creating an invalid MetaAddress
  ///
  /// \note Prefer MetaAddress::invalid()
  constexpr explicit MetaAddress() : PlainMetaAddress({}) {}

  /// Public constructor allowing to create a custom instance to validate
  ///
  /// \note Prefer MetaAddress:fromPC or MetaAddress::fromGeneric
  constexpr explicit MetaAddress(uint64_t Address,
                                 MetaAddressType::Values Type,
                                 uint32_t Epoch = 0,
                                 uint16_t AddressSpace = 0) :
    PlainMetaAddress({ Epoch, AddressSpace, Type, Address }) {

    // Verify the given data
    validate();
  }

  /// @}

public:
  /// \name Factory methods
  ///
  /// @{

  /// Create an invalid MetaAddress
  static constexpr MetaAddress invalid() { return MetaAddress(); }

  /// Create a MetaAddress from a pointer to \p Arch code
  static constexpr MetaAddress fromPC(llvm::Triple::ArchType Arch,
                                      uint64_t PC,
                                      uint32_t Epoch = 0,
                                      uint16_t AddressSpace = 0) {

    // Create the base MetaAddress, it points to code at zero
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

  static MetaAddress fromPC(uint64_t Address, const Features &Features) {
    return MetaAddress::fromPC(Features.Architecture,
                               Address,
                               Features.Epoch,
                               Features.AddressSpace);
  }

  /// Create a generic MetaAddress for architecture \p Arch
  static constexpr MetaAddress fromGeneric(llvm::Triple::ArchType Arch,
                                           uint64_t Address,
                                           uint32_t Epoch = 0,
                                           uint16_t AddressSpace = 0) {
    return MetaAddress(Address,
                       MetaAddressType::genericFromArch(Arch),
                       Epoch,
                       AddressSpace);
  }

  static constexpr MetaAddress fromGeneric(uint64_t Address,
                                           Features Features) {
    return MetaAddress(Address,
                       MetaAddressType::genericFromArch(Features.Architecture),
                       Features.Epoch,
                       Features.AddressSpace);
  }

  /// @}

public:
  /// Deserialize a MetaAddress from an llvm::Constant
  static MetaAddress fromValue(llvm::Value *V);

  /// Serialize a MetaAddress to an llvm::Constant
  llvm::Constant *toValue(llvm::Module *M) const;

public:
  static llvm::Instruction *composeIntegerPC(llvm::IRBuilderBase &B,
                                             llvm::Value *AddressValue,
                                             llvm::Value *EpochValue,
                                             llvm::Value *AddressSpaceValue,
                                             llvm::Value *TypeValue);

  static MetaAddress decomposeIntegerPC(llvm::ConstantInt *Value);
  static MetaAddress decomposeIntegerPC(const llvm::APInt &Value);

public:
  /// If isCode(), let this decay to the corresponding generic address
  constexpr MetaAddress toGeneric() const {
    revng_check(isValid());

    MetaAddress Result = *this;
    Result.Type = MetaAddressType::toGeneric(type());
    return Result;
  }

  constexpr MetaAddress toGeneric64() const {
    revng_check(isValid());

    MetaAddress Result = *this;
    Result.Type = MetaAddressType::Generic64;
    return Result;
  }

  constexpr MetaAddress toPC(llvm::Triple::ArchType Arch) const {
    return fromPC(Arch, Address, Epoch, AddressSpace);
  }

  Features features() const {
    return Features(*MetaAddressType::arch(type()), Epoch, AddressSpace);
  }

public:
  /// @{
  constexpr bool operator==(const MetaAddress &Other) const {
    return tie() == Other.tie();
  }

  constexpr bool operator!=(const MetaAddress &Other) const {
    return not(*this == Other);
  }

  constexpr bool operator<(const MetaAddress &Other) const {
    return tie() < Other.tie();
  }
  constexpr bool operator<=(const MetaAddress &Other) const {
    return tie() <= Other.tie();
  }
  constexpr bool operator>(const MetaAddress &Other) const {
    return tie() > Other.tie();
  }
  constexpr bool operator>=(const MetaAddress &Other) const {
    return tie() >= Other.tie();
  }

  /// @}

  /// \name Address comparisons
  ///
  /// Comparison operators ignoring the MetaAddress type
  ///
  /// @{

  constexpr bool addressEquals(const MetaAddress &Other) const {
    revng_assert(isValid());
    revng_assert(Other.isValid());
    return toGeneric64() == Other.toGeneric64();
  }

  constexpr bool addressDiffers(const MetaAddress &Other) const {
    revng_assert(isValid());
    revng_assert(Other.isValid());
    return !addressEquals(Other);
  }

  constexpr bool addressLowerThan(const MetaAddress &Other) const {
    revng_assert(isValid());
    revng_assert(Other.isValid());
    return toGeneric64() < Other.toGeneric64();
  }

  constexpr bool addressLowerThanOrEqual(const MetaAddress &Other) const {
    revng_assert(isValid());
    revng_assert(Other.isValid());
    return toGeneric64() <= Other.toGeneric64();
  }

  constexpr bool addressGreaterThanOrEqual(const MetaAddress &Other) const {
    revng_assert(isValid());
    revng_assert(Other.isValid());
    return not(addressLowerThan(Other));
  }

  constexpr bool addressGreaterThan(const MetaAddress &Other) const {
    revng_assert(isValid());
    revng_assert(Other.isValid());
    return not(addressLowerThanOrEqual(Other));
  }

  std::optional<uint64_t> operator-(const MetaAddress &Other) const {
    revng_assert(isValid());
    revng_assert(Other.isValid());
    revng_assert(Epoch == Other.Epoch);
    revng_assert(AddressSpace == Other.AddressSpace);
    return (OverflowSafeInt(Address) - Other.Address).value();
  }
  /// @}

  /// \name Arithmetic additions/subtractions
  ///
  /// @{

  template<std::integral T>
  MetaAddress &operator+=(T Offset) {
    if (isInvalid())
      return *this;

    auto Update = [this, Offset](auto NewAddress) {
      if constexpr (std::is_signed_v<T>) {
        if (Offset >= 0)
          NewAddress += Offset;
        else
          NewAddress -= -Offset;
      } else {
        NewAddress += Offset;
      }

      if (NewAddress)
        setAddress(*NewAddress);
      else
        *this = MetaAddress::invalid();
    };

    if (bitSize() == 32)
      Update(OverflowSafeInt<uint32_t>(Address));
    else
      Update(OverflowSafeInt<uint64_t>(Address));

    return *this;
  }

  template<std::integral T>
  MetaAddress &operator-=(T Offset) {
    if (isInvalid())
      return *this;

    auto Update = [this, Offset](auto NewAddress) {
      if constexpr (std::is_signed_v<T>) {
        if (Offset >= 0)
          NewAddress -= Offset;
        else
          NewAddress += -Offset;
      } else {
        NewAddress -= Offset;
      }

      if (NewAddress)
        setAddress(*NewAddress);
      else
        *this = MetaAddress::invalid();
    };

    if (bitSize() == 32)
      Update(OverflowSafeInt<uint32_t>(Address));
    else
      Update(OverflowSafeInt<uint64_t>(Address));

    return *this;
  }

  template<std::integral T>
  MetaAddress operator+(T Offset) const {
    MetaAddress Result = *this;
    Result += Offset;
    return Result;
  }

  template<std::integral T>
  MetaAddress operator-(T Offset) const {
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
  constexpr MetaAddress replaceAddress(uint64_t Address) const {
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
  constexpr uint64_t address() const {
    revng_assert(isValid());
    return Address;
  }

  /// Return the wrapped address in its PC representation
  ///
  /// \note Don't call this method if `!(isValid() && isCode())`
  constexpr uint64_t asPC() const {
    revng_check(isValid());
    return asPCOrZero();
  }

  /// Return the wrapped address in its PC representation, or 0 if invalid
  constexpr uint64_t asPCOrZero() const {
    revng_check(isCode() or isInvalid());

    switch (type()) {
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
    case MetaAddressType::Code_hexagon:
      return Address;

    case MetaAddressType::Generic32:
    case MetaAddressType::Generic64:
      revng_abort();
    }

    revng_abort();
  }

  constexpr uint16_t addressSpace() const {
    revng_check(isValid());
    return AddressSpace;
  }
  constexpr bool isDefaultAddressSpace() const { return addressSpace() == 0; }

  constexpr uint32_t epoch() const {
    revng_check(isValid());
    return Epoch;
  }
  constexpr bool isDefaultEpoch() const { return epoch() == 0; }

  constexpr MetaAddressType::Values type() const {
    return MetaAddressType::Values(Type);
  }
  constexpr bool isInvalid() const {
    return type() == MetaAddressType::Invalid;
  }
  constexpr bool isValid() const { return not isInvalid(); }
  constexpr bool isCode() const { return MetaAddressType::isCode(type()); }
  constexpr bool isCode(llvm::Triple::ArchType Arch) const {
    return MetaAddressType::isCode(type(), Arch);
  }
  constexpr bool isGeneric() const {
    return MetaAddressType::isGeneric(type());
  }
  constexpr unsigned bitSize() const {
    return MetaAddressType::bitSize(type());
  }
  constexpr unsigned alignment() const {
    return MetaAddressType::alignment(type());
  }

  std::optional<llvm::Triple::ArchType> arch() const {
    return MetaAddressType::arch(type());
  }

  constexpr bool isDefaultCode() const {
    return MetaAddressType::isDefaultCode(type());
  }

  /// @}

public:
  constexpr bool isIn(const MetaAddress &Start, const MetaAddress &End) const {
    return Start <= *this and *this < End;
  }

public:
  void dump() const debug_function { dump(dbg); }

  template<typename T>
  void dump(T &Output) const {
    Output << toString();
  }

  template<typename T>
  void dumpRelativeTo(T &Output,
                      const MetaAddress &Base,
                      llvm::StringRef BaseName) const {
    Output << BaseName.data();

    if (Base == *this)
      return;

    Output << ".";

    auto MaybeDifference = *this - Base;
    if (not MaybeDifference)
      Output << MetaAddress::invalid().toString();
    else
      Output << "0x" << llvm::Twine::utohexstr(*MaybeDifference).str();
  }

public:
  constexpr MetaAddress pageStart() const {
    revng_check(isValid());
    return toGeneric() - (Address % 4096);
  }

  MetaAddress nextPageStart() const {
    revng_check(isValid());

    auto Addend = ((OverflowSafeInt(Address) + (4096 - 1)) / 4096) * 4096
                  - Address;

    if (not Addend)
      return MetaAddress::invalid();

    return toGeneric() + *Addend;
  }

private:
  constexpr bool verify() const debug_function {
    // Invalid addresses are all the same
    if (type() == MetaAddressType::Invalid) {
      return *this == invalid();
    }

    if (static_cast<uint16_t>(Type) > MetaAddressType::Code_hexagon)
      return false;

    // Check alignment
    if (Address % alignment() != 0)
      return false;

    // Check address mask
    if (Address != (Address & addressMask()))
      return false;

    return true;
  }

  constexpr void validate() {
    if (not verify())
      setInvalid();
  }

  constexpr void setInvalid() { *this = MetaAddress(); }

  constexpr uint64_t addressMask() const {
    return MetaAddressType::addressMask(type());
  }

  constexpr void setPC(uint64_t PC) {
    if (type() == MetaAddressType::Code_arm_thumb) {

      if ((PC & 1) == 0) {
        setInvalid();
        return;
      }

      PC = PC & ~1;
    }

    setAddress(PC);
  }

  constexpr void setAddress(uint64_t NewAddress) {
    Address = NewAddress & addressMask();
    validate();
  }

public:
  /// \param Arch specifying the "expected" architecture omits it from
  ///        the serialized string. But it also leads to inability
  ///        to deserialize it! So only use if you know what you're doing.
  std::string toString(std::optional<llvm::Triple::ArchType> Arch = {}) const;
  static MetaAddress fromString(llvm::StringRef Text);

private:
  using Tied = std::tuple<const uint32_t &,
                          const uint16_t &,
                          const uint16_t &,
                          const uint64_t &>;
  constexpr Tied tie() const {
    return std::tie(Epoch, AddressSpace, Type, Address);
  }
};

static_assert(sizeof(MetaAddress) <= 128 / 8,
              "MetaAddress is larger than 128 bits");

template<typename T>
struct CompareAddress {};

template<>
struct CompareAddress<MetaAddress> {
  bool operator()(const MetaAddress &LHS, const MetaAddress &RHS) const {
    return LHS.addressLowerThan(RHS);
  }
};

template<>
struct KeyedObjectTraits<MetaAddress>
  : public IdentityKeyedObjectTraits<MetaAddress> {};

namespace std {
template<>
class hash<MetaAddress> {
public:
  uint64_t operator()(const MetaAddress &Address) const {
    return hash_combine(Address.arch(),
                        Address.address(),
                        Address.epoch(),
                        Address.addressSpace());
  }
};
} // namespace std
