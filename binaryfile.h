#ifndef _BINARYFILE_H
#define _BINARYFILE_H

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// Standard includes
#include <set>
#include <string>
#include <vector>

// LLVM includes
#include "llvm/ADT/Optional.h"
#include "llvm/Object/ELFTypes.h"
#include "llvm/Object/Binary.h"

// Local includes
#include "revamb.h"

namespace llvm {
namespace object {
class ObjectFile;
}
}

/// \brief Simple data structure to describe an ELF segment
// TODO: information hiding
struct SegmentInfo {
  /// Produce a name for this segment suitable for human understanding
  std::string generateName();

  llvm::GlobalVariable *Variable; ///< \brief LLVM variable containing this
                                  ///  segment's data
  uint64_t StartVirtualAddress;
  uint64_t EndVirtualAddress;
  bool IsWriteable;
  bool IsExecutable;
  bool IsReadable;
  std::vector<std::pair<uint64_t, uint64_t>> ExecutableSections;
  llvm::ArrayRef<uint8_t> Data;

  bool contains(uint64_t Address) const {
    return StartVirtualAddress <= Address && Address < EndVirtualAddress;
  }

  bool contains(uint64_t Start, uint64_t Size) const {
    return contains(Start) && contains(Start + Size - 1);
  }

  uint64_t size() const { return EndVirtualAddress - StartVirtualAddress; }

  template<class C>
  void insertExecutableRanges(std::back_insert_iterator<C> Inserter) const {
    if (!IsExecutable)
      return;

    if (ExecutableSections.size() > 0) {
      std::copy(ExecutableSections.begin(),
                ExecutableSections.end(),
                Inserter);
    } else {
      Inserter = std::make_pair(StartVirtualAddress, EndVirtualAddress);
    }
  }

};

/// \brief Simple data structure to describe a symbol in an image format
///        independent way
// TODO: information hiding
struct SymbolInfo {
  llvm::StringRef Name;
  uint64_t Address;
  uint64_t Size;

  bool operator<(const SymbolInfo &Other) const {
    return Address < Other.Address;
  }

  bool operator==(const SymbolInfo &Other) const {
    return Name == Other.Name && Address == Other.Address && Size == Other.Size;
  }
};

//
// What follows is a set of functions we use to read an integer of a specified
// (or pointer) size using the appropriate endianess associated to an ELF type.
//

template<typename T, typename EE>
struct Endianess {
  /// \brief Reads an integer of type T, using the endianess of the ELF type EE
  static uint64_t read(const uint8_t *Buf);
};

template<typename T>
struct Endianess<T, llvm::object::ELF32LE> {
  static uint64_t read(const uint8_t *Buf) {
    return llvm::support::endian::read<T,
                                       llvm::support::little,
                                       llvm::support::unaligned>(Buf);
  }
};

template<typename T>
struct Endianess<T, llvm::object::ELF64LE> {
  static uint64_t read(const uint8_t *Buf) {
    return llvm::support::endian::read<T,
                                       llvm::support::little,
                                       llvm::support::unaligned>(Buf);
  }
};

template<typename T>
struct Endianess<T, llvm::object::ELF32BE> {
  static uint64_t read(const uint8_t *Buf) {
    return llvm::support::endian::read<T,
                                       llvm::support::big,
                                       llvm::support::unaligned>(Buf);
  }
};

template<typename T>
struct Endianess<T, llvm::object::ELF64BE> {
  static uint64_t read(const uint8_t *Buf) {
    return llvm::support::endian::read<T,
                                       llvm::support::big,
                                       llvm::support::unaligned>(Buf);
  }
};

/// \brief Read a pointer-sized integer according to the given ELF type EE
template<typename EE>
static inline uint64_t readPointer(const uint8_t *Buf);

template<>
inline uint64_t readPointer<llvm::object::ELF32LE>(const uint8_t *Buf) {
  return Endianess<uint32_t, llvm::object::ELF32LE>::read(Buf);
}

template<>
inline uint64_t readPointer<llvm::object::ELF32BE>(const uint8_t *Buf) {
  return Endianess<uint32_t, llvm::object::ELF32BE>::read(Buf);
}

template<>
inline uint64_t readPointer<llvm::object::ELF64LE>(const uint8_t *Buf) {
  return Endianess<uint64_t, llvm::object::ELF64LE>::read(Buf);
}

template<>
inline uint64_t readPointer<llvm::object::ELF64BE>(const uint8_t *Buf) {
  return Endianess<uint64_t, llvm::object::ELF64BE>::read(Buf);
}

/// \brief A pair on steroids to wrap a value or a pointer to a value
class Pointer {
public:
  Pointer() { }

  Pointer(bool IsIndirect, uint64_t Value) :
    IsIndirect(IsIndirect),
    Value(Value) { }

  bool isIndirect() const { return IsIndirect; }
  uint64_t value() const { return Value; }

private:
  bool IsIndirect;
  uint64_t Value;

};

/// \brief BinaryFile describes an input image file in a semi-architecture
///        independent way
class BinaryFile {
public:
  /// \param FilePath the path to the input file.
  /// \param UseSections whether information in sections, if available, should
  ///        be employed or not. This is useful to precisely identify exeutable
  ///        code.
  BinaryFile(std::string FilePath, bool UseSections);

  llvm::Optional<llvm::ArrayRef<uint8_t>>
  getAddressData(uint64_t Address) const {
    for (const SegmentInfo &Segment : Segments) {
      if (Segment.contains(Address)) {
        uint64_t Offset = Address - Segment.StartVirtualAddress;
        uint64_t Size = Segment.size() - Offset;
        return { llvm::ArrayRef<uint8_t>(Segment.Data.data() + Offset, Size) };
      }
    }

    return llvm::Optional<llvm::ArrayRef<uint8_t>>();
  }

  //
  // Accessor methods
  //

  const Architecture &architecture() const { return TheArchitecture; }
  std::vector<SegmentInfo> &segments() { return Segments; }
  const std::vector<SegmentInfo> &segments() const { return Segments; }
  const std::vector<SymbolInfo> &symbols() const { return Symbols; }
  const std::set<uint64_t> &landingPads() const { return LandingPads; }
  uint64_t entryPoint() const { return EntryPoint; }

  //
  // ELF specific accessors
  //

  uint64_t programHeadersAddress() const { return ProgramHeaders.Address; }
  unsigned programHeaderSize() const { return ProgramHeaders.Size; }
  unsigned programHeadersCount() const { return ProgramHeaders.Count; }

  /// \brief Gets the actual value of a Pointer object, possibly reading it from
  ///        memory
  template<typename T>
  uint64_t getPointer(Pointer Ptr) const {
    if (!Ptr.isIndirect())
      return Ptr.value();

    auto R = getAddressData(Ptr.value());
    assert(R && "Pointer not available in any segment");
    llvm::ArrayRef<uint8_t> Pointer = *R;

    return ::readPointer<T>(Pointer.data());
  }

private:
  //
  // ELF-specific methods
  //

  /// \brief Parse an ELF file to load all the required information
  template<typename T>
  void parseELF(llvm::object::ObjectFile *TheBinary, bool UseSections);

  /// \brief Parse the .eh_frame_hdr section to obtain the address and the
  ///        number of FDEs in .eh_frame
  ///
  /// \return a pair containing the pointer to the .eh_frame section and the
  ///         count of FDEs in the .eh_frame_hdr section (which should match the
  ///         number of FDEs in .eh_frame)
  template<typename T>
  std::pair<uint64_t, uint64_t>
  ehFrameFromEhFrameHdr(uint64_t EHFrameHdrAddress);

  /// \brief Parse the .eh_frame section to collect all the landing pads
  ///
  /// \param EHFrameAddress the address of the .eh_frame section
  /// \param FDEsCount the count of FDEs in the .eh_frame section
  /// \param EHFrameSize the size of the .eh_frame section
  ///
  /// \note Either \p FDEsCount or \p EHFrameSize have to be specified
  template<typename T>
  void parseEHFrame(uint64_t EHFrameAddress,
                    llvm::Optional<uint64_t> FDEsCount,
                    llvm::Optional<uint64_t> EHFrameSize);

  /// \brief Parse an LSDA to collect its landing pads
  ///
  /// \param FDEStart the start address of the FDE to which this LSDA is
  ///        associated
  /// \param LSDAAddress the address of the target LSDA
  template<typename T>
  void parseLSDA(uint64_t FDEStart, uint64_t LSDAAddress);

private:
  llvm::object::OwningBinary<llvm::object::Binary> BinaryHandle;
  Architecture TheArchitecture;
  std::vector<SymbolInfo> Symbols;
  std::vector<SegmentInfo> Segments;
  std::set<uint64_t> LandingPads; ///< the set of the landing pad addresses
                                  ///  collected from .eh_frame

  uint64_t EntryPoint; ///< the program's entry point

  //
  // ELF specific fields
  //

  struct {
    uint64_t Address;
    unsigned Count;
    unsigned Size;
  } ProgramHeaders;
};

#endif // _BINARYFILE_H
