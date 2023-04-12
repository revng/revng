#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <cstdint>
#include <limits>
#include <type_traits>

#include "llvm/BinaryFormat/Dwarf.h"
#include "llvm/Object/ELFObjectFile.h"
#include "llvm/Support/Endian.h"
#include "llvm/Support/LEB128.h"

#include "revng/Support/Assert.h"
#include "revng/Support/MetaAddress.h"

//
// What follows is a set of functions we use to read an integer of a specified
// (or pointer) size using the appropriate endianness associated to an ELF type.
//

template<typename T, typename EE>
struct Endianness {
  /// Reads an integer of type T, using the endianness of the ELF type EE
  static uint64_t read(const uint8_t *Buf);
};

template<typename T>
struct Endianness<T, llvm::object::ELF32LE> {
  static uint64_t read(const uint8_t *Buf) {
    using namespace llvm::support;
    return llvm::support::endian::read<T, little, unaligned>(Buf);
  }
};

template<typename T>
struct Endianness<T, llvm::object::ELF64LE> {
  static uint64_t read(const uint8_t *Buf) {
    using namespace llvm::support;
    return llvm::support::endian::read<T, little, unaligned>(Buf);
  }
};

template<typename T>
struct Endianness<T, llvm::object::ELF32BE> {
  static uint64_t read(const uint8_t *Buf) {
    using namespace llvm::support;
    return llvm::support::endian::read<T, big, unaligned>(Buf);
  }
};

template<typename T>
struct Endianness<T, llvm::object::ELF64BE> {
  static uint64_t read(const uint8_t *Buf) {
    using namespace llvm::support;
    return llvm::support::endian::read<T, big, unaligned>(Buf);
  }
};

/// Read a pointer-sized integer according to the given ELF type EE
template<typename EE>
inline uint64_t readPointer(const uint8_t *Buf);

template<>
inline uint64_t readPointer<llvm::object::ELF32LE>(const uint8_t *Buf) {
  return Endianness<uint32_t, llvm::object::ELF32LE>::read(Buf);
}

template<>
inline uint64_t readPointer<llvm::object::ELF32BE>(const uint8_t *Buf) {
  return Endianness<uint32_t, llvm::object::ELF32BE>::read(Buf);
}

template<>
inline uint64_t readPointer<llvm::object::ELF64LE>(const uint8_t *Buf) {
  return Endianness<uint64_t, llvm::object::ELF64LE>::read(Buf);
}

template<>
inline uint64_t readPointer<llvm::object::ELF64BE>(const uint8_t *Buf) {
  return Endianness<uint64_t, llvm::object::ELF64BE>::read(Buf);
}

/// A pair on steroids to wrap a value or a pointer to a value
class Pointer {
public:
  Pointer() : IsIndirect(false), Value(MetaAddress::invalid()) {}

  Pointer(bool IsIndirect, MetaAddress Value) :
    IsIndirect(IsIndirect), Value(Value) {}

  bool isIndirect() const { return IsIndirect; }
  MetaAddress value() const { return Value; }

private:
  bool IsIndirect;
  MetaAddress Value;
};

template<typename E>
class DwarfReader {
public:
  DwarfReader(llvm::Triple::ArchType Architecture,
              llvm::ArrayRef<uint8_t> Buffer,
              MetaAddress Address) :
    Architecture(Architecture),
    Address(Address),
    Start(Buffer.data()),
    Cursor(Buffer.data()),
    End(Buffer.data() + Buffer.size()) {}

  uint8_t readNextU8() { return readNext<uint8_t>(); }
  uint16_t readNextU16() { return readNext<uint16_t>(); }
  uint32_t readNextU32() { return readNext<uint32_t>(); }
  uint64_t readNextU64() { return readNext<uint64_t>(); }
  uint64_t readNextU() {
    if (is64())
      return readNextU64();
    else
      return readNextU32();
  }

  uint64_t readULEB128() {
    unsigned Length;
    uint64_t Result = llvm::decodeULEB128(Cursor, &Length);
    Cursor += Length;
    revng_assert(Cursor <= End);
    return Result;
  }

  int64_t readSLEB128() {
    unsigned Length;
    int64_t Result = llvm::decodeSLEB128(Cursor, &Length);
    Cursor += Length;
    revng_assert(Cursor <= End);
    return Result;
  }

  int64_t readSignedValue(unsigned Encoding) {
    return static_cast<int64_t>(readValue(Encoding));
  }

  uint64_t readUnsignedValue(unsigned Encoding) {
    return static_cast<uint64_t>(readValue(Encoding));
  }

  Pointer
  readPointer(unsigned Encoding, MetaAddress Base = MetaAddress::invalid()) {
    using namespace llvm;
    revng_assert((Encoding & ~(0x70 | 0x0F | dwarf::DW_EH_PE_indirect)) == 0);

    // Handle PC-relative values
    revng_assert(Cursor >= Start);
    if ((Encoding & 0x70) == dwarf::DW_EH_PE_pcrel) {
      revng_assert(Base.isInvalid());
      Base = Address + (Cursor - Start);
    }

    if (isSigned(Encoding & 0x0F)) {
      return readPointerInternal(readSignedValue(Encoding), Encoding, Base);
    } else {
      return readPointerInternal(readUnsignedValue(Encoding), Encoding, Base);
    }
  }

  void moveTo(uint64_t Offset) {
    const uint8_t *NewCursor = Start + Offset;
    revng_assert(NewCursor >= Cursor && NewCursor <= End);
    Cursor = NewCursor;
  }

  bool eof() const { return Cursor >= End; }
  uint64_t offset() const { return Cursor - Start; }

private:
  template<typename T>
  std::conditional_t<std::numeric_limits<T>::is_signed, int64_t, uint64_t>
  readNext() {
    using namespace llvm;
    constexpr bool IsSigned = std::numeric_limits<T>::is_signed;
    using ReturnType = std::conditional_t<IsSigned, int64_t, uint64_t>;
    revng_assert(Cursor + sizeof(T) <= End);
    auto Result = static_cast<T>(Endianness<T, E>::read(Cursor));
    Cursor += sizeof(T);
    return static_cast<ReturnType>(Result);
  }

  static bool isSigned(unsigned Format) {
    using namespace llvm;

    switch (Format) {
    case dwarf::DW_EH_PE_sleb128:
    case dwarf::DW_EH_PE_signed:
    case dwarf::DW_EH_PE_sdata2:
    case dwarf::DW_EH_PE_sdata4:
    case dwarf::DW_EH_PE_sdata8:
      return true;
    case dwarf::DW_EH_PE_absptr:
    case dwarf::DW_EH_PE_uleb128:
    case dwarf::DW_EH_PE_udata2:
    case dwarf::DW_EH_PE_udata4:
    case dwarf::DW_EH_PE_udata8:
      return false;
    default:
      revng_abort("Unknown Encoding");
    }
  }

  uint64_t readValue(unsigned Encoding) {
    using namespace llvm;

    revng_assert((Encoding & ~(0x70 | 0x0F | dwarf::DW_EH_PE_indirect)) == 0);

    // Extract the format
    unsigned Format = Encoding & 0x0F;
    switch (Format) {
    case dwarf::DW_EH_PE_uleb128:
      return readULEB128();
    case dwarf::DW_EH_PE_sleb128:
      return readSLEB128();
    case dwarf::DW_EH_PE_absptr:
      if (is64())
        return readNext<uint64_t>();
      else
        return readNext<uint32_t>();
    case dwarf::DW_EH_PE_signed:
      if (is64())
        return readNext<int64_t>();
      else
        return readNext<int32_t>();
    case dwarf::DW_EH_PE_udata2:
      return readNext<uint16_t>();
    case dwarf::DW_EH_PE_sdata2:
      return readNext<int16_t>();
    case dwarf::DW_EH_PE_udata4:
      return readNext<uint32_t>();
    case dwarf::DW_EH_PE_sdata4:
      return readNext<int32_t>();
    case dwarf::DW_EH_PE_udata8:
      return readNext<uint64_t>();
    case dwarf::DW_EH_PE_sdata8:
      return readNext<int64_t>();
    default:
      revng_unreachable("Unknown Encoding");
    }
  }

  template<typename T>
  Pointer readPointerInternal(T Value, unsigned Encoding, MetaAddress Base) {
    using namespace llvm;

    bool IsIndirect = Encoding & dwarf::DW_EH_PE_indirect;

    if (Base.isInvalid()) {
      return Pointer(IsIndirect, MetaAddress::fromGeneric(Architecture, Value));
    } else {
      unsigned EncodingRelative = Encoding & 0x70;
      revng_assert(EncodingRelative == 0 || EncodingRelative == 0x10);
      return Pointer(IsIndirect, Base + Value);
    }
  }

  bool is64() const;

private:
  llvm::Triple::ArchType Architecture;
  MetaAddress Address;
  const uint8_t *Start;
  const uint8_t *Cursor;
  const uint8_t *End;
};

template<>
inline bool DwarfReader<llvm::object::ELF32BE>::is64() const {
  return false;
}
template<>
inline bool DwarfReader<llvm::object::ELF32LE>::is64() const {
  return false;
}
template<>
inline bool DwarfReader<llvm::object::ELF64BE>::is64() const {
  return true;
}
template<>
inline bool DwarfReader<llvm::object::ELF64LE>::is64() const {
  return true;
}
