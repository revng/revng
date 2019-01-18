#ifndef BINARYFILE_H
#define BINARYFILE_H

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// Standard includes
#include <set>
#include <string>
#include <vector>

// Boost includes
#include <boost/icl/interval_map.hpp>

// LLVM includes
#include "llvm/ADT/Optional.h"
#include "llvm/Object/Binary.h"
#include "llvm/Object/ELFTypes.h"

// Local libraries includes
#include "revng/Support/revng.h"

namespace llvm {
namespace object {
class ObjectFile;
}
} // namespace llvm

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
      std::copy(ExecutableSections.begin(), ExecutableSections.end(), Inserter);
    } else {
      Inserter = std::make_pair(StartVirtualAddress, EndVirtualAddress);
    }
  }
};

namespace LabelType {

enum Values {
  Invalid,
  AbsoluteValue,
  BaseRelativeValue,
  SymbolRelativeValue,
  Symbol
};

inline const char *getName(Values V) {
  switch (V) {
  case Invalid:
    return "Invalid";
  case AbsoluteValue:
    return "AbsoluteValue";
  case BaseRelativeValue:
    return "BaseRelativeValue";
  case SymbolRelativeValue:
    return "SymbolRelativeValue";
  case Symbol:
    return "Symbol";
  }

  revng_abort();
}

} // namespace LabelType

namespace SymbolType {

enum Values { Unknown, Code, Data, Section, File };

inline const char *getName(Values V) {
  switch (V) {
  case Unknown:
    return "Unknown";
  case Code:
    return "Code";
  case Data:
    return "Data";
  case Section:
    return "Section";
  case File:
    return "File";
  }

  revng_abort();
}

inline SymbolType::Values fromELF(unsigned char ELFSymbolType) {
  switch (ELFSymbolType) {
  case llvm::ELF::STT_NOTYPE:
  case llvm::ELF::STT_TLS:
    return SymbolType::Unknown;

  case llvm::ELF::STT_FUNC:
    return SymbolType::Code;

  case llvm::ELF::STT_OBJECT:
    return SymbolType::Data;

  case llvm::ELF::STT_SECTION:
    return SymbolType::Section;

  case llvm::ELF::STT_FILE:
    return SymbolType::File;

  default:
    revng_abort("Unexpected symbol type");
  }
}

} // namespace SymbolType

namespace LabelOrigin {

enum Values { Unknown, StaticSymbol, DynamicSymbol, DynamicRelocation };

inline const char *getName(Values V) {

  switch (V) {
  case Unknown:
    return "Unknown";
  case StaticSymbol:
    return "StaticSymbol";
  case DynamicSymbol:
    return "DynamicSymbol";
  case DynamicRelocation:
    return "DynamicRelocation";
  }

  revng_abort();
}

} // namespace LabelOrigin

class Label {
private:
  LabelType::Values Type;
  uint64_t Address;
  uint64_t Size;

  /// Name of the symbol, if any
  llvm::StringRef SymbolName;
  SymbolType::Values SymbolType;

  /// Label value. It has differente meanings depending on the label type
  uint64_t Value;

  LabelOrigin::Values Origin;
  bool SizeIsVirtual;

private:
  Label(LabelOrigin::Values Origin, uint64_t Address, uint64_t Size) :
    Type(LabelType::Invalid),
    Address(Address),
    Size(Size),
    SymbolName(),
    SymbolType(SymbolType::Unknown),
    Value(0),
    Origin(Origin),
    SizeIsVirtual(false) {}

public:
  static Label createInvalid() { return Label(LabelOrigin::Unknown, 0, 0); }

  static Label createAbsoluteValue(LabelOrigin::Values Origin,
                                   uint64_t Address,
                                   uint64_t Size,
                                   uint64_t Value) {
    Label Result(Origin, Address, Size);
    Result.Type = LabelType::AbsoluteValue;
    Result.Value = Value;
    return Result;
  }

  static Label createBaseRelativeValue(LabelOrigin::Values Origin,
                                       uint64_t Address,
                                       uint64_t Size,
                                       uint64_t Value) {
    Label Result(Origin, Address, Size);
    Result.Type = LabelType::BaseRelativeValue;
    Result.Value = Value;
    return Result;
  }

  static Label createSymbolRelativeValue(LabelOrigin::Values Origin,
                                         uint64_t Address,
                                         uint64_t Size,
                                         llvm::StringRef SymbolName,
                                         SymbolType::Values SymbolType,
                                         uint64_t Offset) {
    Label Result(Origin, Address, Size);
    Result.Type = LabelType::SymbolRelativeValue;
    Result.SymbolName = SymbolName;
    Result.SymbolType = SymbolType;
    Result.Value = Offset;
    return Result;
  }

  static Label createSymbol(LabelOrigin::Values Origin,
                            uint64_t Address,
                            uint64_t Size,
                            llvm::StringRef SymbolName,
                            SymbolType::Values SymbolType) {
    Label Result(Origin, Address, Size);
    Result.Type = LabelType::Symbol;
    Result.SymbolName = SymbolName;
    Result.SymbolType = SymbolType;
    return Result;
  }

public:
  bool operator==(const Label &Other) const {
    const auto &LHS = std::tie(Origin,
                               Type,
                               Address,
                               Size,
                               SymbolName,
                               Value,
                               SymbolType);
    const auto &RHS = std::tie(Other.Origin,
                               Other.Type,
                               Other.Address,
                               Other.Size,
                               Other.SymbolName,
                               Other.Value,
                               Other.SymbolType);
    return LHS == RHS;
  }

  bool operator<(const Label &Other) const {
    const auto &LHS = std::tie(Origin,
                               Type,
                               Address,
                               Size,
                               SymbolName,
                               Value,
                               SymbolType);
    const auto &RHS = std::tie(Other.Origin,
                               Other.Type,
                               Other.Address,
                               Other.Size,
                               Other.SymbolName,
                               Other.Value,
                               Other.SymbolType);
    return LHS < RHS;
  }

public:
  LabelType::Values type() const { return Type; }

  bool isInvalid() const { return Type == LabelType::Invalid; }
  bool isAbsoluteValue() const { return Type == LabelType::AbsoluteValue; }
  bool isBaseRelativeValue() const {
    return Type == LabelType::BaseRelativeValue;
  }
  bool isSymbolRelativeValue() const {
    return Type == LabelType::SymbolRelativeValue;
  }
  bool isSymbol() const { return Type == LabelType::Symbol; }

  bool isCode() const { return SymbolType == SymbolType::Code; }

  uint64_t address() const { return Address; }
  uint64_t size() const { return Size; }

  uint64_t value() const {
    revng_assert(isAbsoluteValue() or isBaseRelativeValue());
    return Value;
  }

  llvm::StringRef symbolName() const {
    revng_assert(isSymbolRelativeValue() or isSymbol());
    return SymbolName;
  }

  uint64_t offset() const {
    revng_assert(isSymbolRelativeValue());
    return Value;
  }

  void setVirtualSize(uint64_t VirtualSize) {
    SizeIsVirtual = true;
    Size = VirtualSize;
  }

  bool isSizeVirtual() const { return SizeIsVirtual; }

  bool matches(uint64_t Address, uint64_t Size) const {
    return this->Address == Address and this->Size == Size;
  }

  bool contains(uint64_t Address, uint64_t Size) const {
    return (this->Address <= Address
            and (Address + Size) <= (this->Address + this->Size));
  }

  void dump() const debug_function {
    dump(dbg);
    dbg << "\n";
  }

  template<typename T>
  void dump(T &Output) const {
    Output << LabelType::getName(Type) << " @ (0x" << std::hex << Address << ","
           << Size << ") ";

    if (isSymbolRelativeValue() or isSymbol())
      Output << SymbolName.data();
    else if (isBaseRelativeValue())
      Output << "IMAGE_BASE";

    if (isSymbolRelativeValue() or isBaseRelativeValue())
      Output << "+";

    if (isSymbolRelativeValue() or isAbsoluteValue() or isBaseRelativeValue())
      Output << "0x" << std::hex << Value;

    if (isSymbolRelativeValue() or isSymbol()) {
      Output << " [" << SymbolType::getName(SymbolType) << "]";
    }

    Output << " [from " << LabelOrigin::getName(Origin) << "]";
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
    using namespace llvm::support;
    return endian::read<T, little, unaligned>(Buf);
  }
};

template<typename T>
struct Endianess<T, llvm::object::ELF64LE> {
  static uint64_t read(const uint8_t *Buf) {
    using namespace llvm::support;
    return endian::read<T, little, unaligned>(Buf);
  }
};

template<typename T>
struct Endianess<T, llvm::object::ELF32BE> {
  static uint64_t read(const uint8_t *Buf) {
    using namespace llvm::support;
    return endian::read<T, big, unaligned>(Buf);
  }
};

template<typename T>
struct Endianess<T, llvm::object::ELF64BE> {
  static uint64_t read(const uint8_t *Buf) {
    using namespace llvm::support;
    return endian::read<T, big, unaligned>(Buf);
  }
};

/// \brief Read a pointer-sized integer according to the given ELF type EE
template<typename EE>
inline uint64_t readPointer(const uint8_t *Buf);

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
  Pointer() {}

  Pointer(bool IsIndirect, uint64_t Value) :
    IsIndirect(IsIndirect),
    Value(Value) {}

  bool isIndirect() const { return IsIndirect; }
  uint64_t value() const { return Value; }

private:
  bool IsIndirect;
  uint64_t Value;
};

class FilePortion;

/// \brief BinaryFile describes an input image file in a semi-architecture
///        independent way
class BinaryFile {
public:
  using LabelList = llvm::SmallVector<Label *, 6u>;
  using LabelIntervalMap = boost::icl::interval_map<uint64_t, LabelList>;

  enum Endianess { OriginalEndianess, BigEndian, LittleEndian };

public:
  /// \param FilePath the path to the input file.
  /// \param UseSections whether information in sections, if available, should
  ///        be employed or not. This is useful to precisely identify exeutable
  ///        code.
  BinaryFile(std::string FilePath, uint64_t BaseAddress);

  llvm::Optional<llvm::ArrayRef<uint8_t>>
  getAddressData(uint64_t Address) const {
    const SegmentInfo *Segment = findSegment(Address);
    if (Segment != nullptr) {
      uint64_t Offset = Address - Segment->StartVirtualAddress;
      uint64_t Size = Segment->size() - Offset;
      return { llvm::ArrayRef<uint8_t>(Segment->Data.data() + Offset, Size) };
    } else {
      return llvm::Optional<llvm::ArrayRef<uint8_t>>();
    }
  }

  //
  // Accessor methods
  //

  const Architecture &architecture() const { return TheArchitecture; }
  std::vector<SegmentInfo> &segments() { return Segments; }
  const std::vector<SegmentInfo> &segments() const { return Segments; }
  const LabelIntervalMap &labels() const { return LabelsMap; }
  const std::set<uint64_t> &landingPads() const { return LandingPads; }
  const std::set<uint64_t> &codePointers() const { return CodePointers; }
  uint64_t entryPoint() const { return EntryPoint; }

  const std::vector<std::string> &neededLibraryNames() const {
    return NeededLibraryNames;
  }

  const std::map<llvm::StringRef, uint64_t> &canonicalValues() const {
    return CanonicalValues;
  }

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
    revng_assert(R, "Pointer not available in any segment");
    llvm::ArrayRef<uint8_t> Pointer = *R;

    return ::readPointer<T>(Pointer.data());
  }

  /// \brief Try to read an integer from the binary
  llvm::Optional<uint64_t> readRawValue(uint64_t Address,
                                        unsigned Size,
                                        Endianess E = OriginalEndianess) const;

  uint64_t relocate(uint64_t Address) const { return BaseAddress + Address; }

private:
  //
  // ELF-specific methods
  //

  /// \brief Parse an ELF file to load all the required information
  template<typename T, bool HasAddend>
  void parseELF(llvm::object::ObjectFile *TheBinary, uint64_t BaseAddress);

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

  /// \brief Compute the symbol count according to the given relocation table
  ///
  /// \return the index of the highest symbol referenced in the relocations,
  ///         plus 1
  template<typename T, bool HasAddend>
  uint64_t symbolsCount(const FilePortion &Relocations);

  /// \brief Process a relocation and produce a Label
  Label parseRelocation(unsigned char RelocationType,
                        uint64_t Target,
                        uint64_t Addend,
                        llvm::StringRef SymbolName,
                        uint64_t SymbolSize,
                        SymbolType::Values SymbolType);

  template<typename T, bool Addend>
  using Elf_Rel_Array = llvm::ArrayRef<llvm::object::Elf_Rel_Impl<T, Addend>>;

  /// \brief Register a label for each input relocation
  template<typename T, bool HasAddend>
  void registerRelocations(Elf_Rel_Array<T, HasAddend> Relocations,
                           const FilePortion &Dynsym,
                           const FilePortion &Dynstr);

  void registerLabel(const Label &NewLabel) {
    if (NewLabel.isInvalid())
      return;

    Labels.push_back(NewLabel);
  }

  void rebuildLabelsMap();

  SegmentInfo *findSegment(uint64_t Address) {
    for (SegmentInfo &Segment : Segments)
      if (Segment.contains(Address))
        return &Segment;
    return nullptr;
  }

  const SegmentInfo *findSegment(uint64_t Address) const {
    for (const SegmentInfo &Segment : Segments)
      if (Segment.contains(Address))
        return &Segment;
    return nullptr;
  }

private:
  llvm::object::OwningBinary<llvm::object::Binary> BinaryHandle;
  Architecture TheArchitecture;
  std::vector<SegmentInfo> Segments;
  std::vector<std::string> NeededLibraryNames;
  std::set<uint64_t> LandingPads; ///< the set of the landing pad addresses
                                  ///  collected from .eh_frame
  std::set<uint64_t> CodePointers; ///< These are taken from dynamic
                                   ///  symbols/relocations.
  std::map<llvm::StringRef, uint64_t> CanonicalValues;
  std::vector<Label> Labels;
  LabelIntervalMap LabelsMap;

  uint64_t EntryPoint; ///< the program's entry point
  uint64_t BaseAddress;

  //
  // ELF specific fields
  //

  struct {
    uint64_t Address;
    unsigned Count;
    unsigned Size;
  } ProgramHeaders;
};

#endif // BINARYFILE_H
