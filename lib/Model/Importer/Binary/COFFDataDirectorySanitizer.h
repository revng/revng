#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <cstring>
#include <memory>

#include "llvm/Object/COFF.h"
#include "llvm/Support/MemoryBuffer.h"

/// Some PE/COFF binaries carry a Load Config Table data directory whose RVA
/// does not fall within any section (e.g. a stale entry left over from an
/// older or non-standard linker that never actually emitted a load config
/// directory). `llvm::object::createBinary` resolves this directory eagerly
/// while constructing the `COFFObjectFile` and fails to open the file at all
/// if it's malformed this way, even though the rest of the binary may
/// otherwise be usable -- see `COFFObjectFile::initLoadConfigPtr()` in
/// `llvm/lib/Object/COFFObjectFile.cpp`.
///
/// Detect this specific situation ahead of calling `createBinary` and, if
/// found, return a patched copy of \p Buffer with the load config directory
/// zeroed out, i.e., treated as absent (which is what happens whenever
/// `RelativeVirtualAddress == 0` anyway). Returns nullptr if no patch is
/// necessary, which includes the case where \p Buffer is not a well-formed
/// PE/COFF file: in that case we leave it untouched and let `createBinary`
/// produce its own diagnostic.
inline std::unique_ptr<llvm::MemoryBuffer>
sanitizeCOFFLoadConfigDirectory(const llvm::MemoryBuffer &Buffer) {
  using namespace llvm;
  using namespace llvm::object;

  const char *Base = Buffer.getBufferStart();
  const uint64_t Size = Buffer.getBufferSize();

  auto InBounds = [Size](uint64_t Offset, uint64_t Length) {
    return Offset <= Size and Length <= Size - Offset;
  };

  if (not InBounds(0, sizeof(dos_header)))
    return nullptr;

  const auto *DOSHeader = reinterpret_cast<const dos_header *>(Base);
  if (DOSHeader->Magic[0] != 'M' or DOSHeader->Magic[1] != 'Z')
    return nullptr;

  uint64_t PEOffset = DOSHeader->AddressOfNewExeHeader;
  if (not InBounds(PEOffset, sizeof(COFF::PEMagic)))
    return nullptr;
  if (std::memcmp(Base + PEOffset, COFF::PEMagic, sizeof(COFF::PEMagic)) != 0)
    return nullptr;

  uint64_t COFFHeaderOffset = PEOffset + sizeof(COFF::PEMagic);
  if (not InBounds(COFFHeaderOffset, sizeof(coff_file_header)))
    return nullptr;

  const auto *COFFHeader = reinterpret_cast<const coff_file_header *>(
    Base + COFFHeaderOffset);
  uint64_t OptionalHeaderOffset = COFFHeaderOffset + sizeof(coff_file_header);
  uint16_t SizeOfOptionalHeader = COFFHeader->SizeOfOptionalHeader;
  if (SizeOfOptionalHeader == 0
      or not InBounds(OptionalHeaderOffset, SizeOfOptionalHeader))
    return nullptr;

  if (not InBounds(OptionalHeaderOffset, sizeof(support::ulittle16_t)))
    return nullptr;
  const auto *MagicPtr = Base + OptionalHeaderOffset;
  uint16_t Magic = *reinterpret_cast<const support::ulittle16_t *>(MagicPtr);

  uint64_t DataDirectoryOffset = 0;
  uint32_t NumberOfRvaAndSize = 0;
  if (Magic == COFF::PE32Header::PE32) {
    if (SizeOfOptionalHeader < sizeof(pe32_header)
        or not InBounds(OptionalHeaderOffset, sizeof(pe32_header)))
      return nullptr;
    const auto *Header = reinterpret_cast<const pe32_header *>(MagicPtr);
    DataDirectoryOffset = OptionalHeaderOffset + sizeof(pe32_header);
    NumberOfRvaAndSize = Header->NumberOfRvaAndSize;
  } else if (Magic == COFF::PE32Header::PE32_PLUS) {
    if (SizeOfOptionalHeader < sizeof(pe32plus_header)
        or not InBounds(OptionalHeaderOffset, sizeof(pe32plus_header)))
      return nullptr;
    const auto *Header = reinterpret_cast<const pe32plus_header *>(MagicPtr);
    DataDirectoryOffset = OptionalHeaderOffset + sizeof(pe32plus_header);
    NumberOfRvaAndSize = Header->NumberOfRvaAndSize;
  } else {
    // Not a recognized PE32/PE32+ optional header: let createBinary deal
    // with it.
    return nullptr;
  }

  // The file doesn't even declare a load config table entry.
  if (COFF::LOAD_CONFIG_TABLE >= NumberOfRvaAndSize)
    return nullptr;

  uint64_t OptionalHeaderEnd = OptionalHeaderOffset + SizeOfOptionalHeader;
  uint64_t LoadConfigEntryOffset = DataDirectoryOffset
                                   + COFF::LOAD_CONFIG_TABLE
                                       * sizeof(data_directory);
  // A contradictory SizeOfOptionalHeader must not make us interpret section
  // table bytes as a data directory entry and patch them in place.
  if (LoadConfigEntryOffset > OptionalHeaderEnd
      or sizeof(data_directory)
           > OptionalHeaderEnd - LoadConfigEntryOffset)
    return nullptr;
  if (not InBounds(LoadConfigEntryOffset, sizeof(data_directory)))
    return nullptr;

  uint64_t SectionTableOffset = OptionalHeaderOffset + SizeOfOptionalHeader;
  uint64_t SectionTableSize = uint64_t(COFFHeader->NumberOfSections)
                              * sizeof(coff_section);
  if (not InBounds(SectionTableOffset, SectionTableSize))
    return nullptr;

  const auto *Sections = reinterpret_cast<const coff_section *>(
    Base + SectionTableOffset);
  uint32_t SectionCount = COFFHeader->NumberOfSections;

  // Mirrors the "is this RVA covered by some section" check performed by
  // COFFObjectFile::getRvaPtr: if it's not, opening the file will fail.
  auto IsCoveredBySection = [&](uint32_t RVA) {
    for (uint32_t I = 0; I < SectionCount; ++I) {
      uint32_t Start = Sections[I].VirtualAddress;
      uint32_t End = Start + Sections[I].VirtualSize;
      if (Start <= RVA and RVA < End)
        return true;
    }
    return false;
  };

  const auto *LoadConfigEntry = reinterpret_cast<const data_directory *>(
    Base + LoadConfigEntryOffset);
  uint32_t RVA = LoadConfigEntry->RelativeVirtualAddress;
  if (RVA == 0 or IsCoveredBySection(RVA))
    return nullptr;

  auto Result = WritableMemoryBuffer::getNewUninitMemBuffer(
    Size, Buffer.getBufferIdentifier());
  if (not Result)
    return nullptr;
  std::memcpy(Result->getBufferStart(), Base, Size);

  auto *PatchedEntry = reinterpret_cast<data_directory *>(
    Result->getBufferStart() + LoadConfigEntryOffset);
  PatchedEntry->RelativeVirtualAddress = 0;
  PatchedEntry->Size = 0;

  return std::move(Result);
}
