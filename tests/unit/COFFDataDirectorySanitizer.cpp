//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <cstdint>
#include <cstring>
#include <optional>
#include <vector>

#define BOOST_TEST_MODULE COFFDataDirectorySanitizer
bool init_unit_test();
#include "boost/test/unit_test.hpp"

#include "llvm/Object/Binary.h"
#include "llvm/Support/MemoryBuffer.h"

#include "revng/UnitTestHelpers/UnitTestHelpers.h"

#include "lib/Model/Importer/Binary/COFFDataDirectorySanitizer.h"

using namespace llvm;
using namespace llvm::object;

// Builds a minimal, otherwise well-formed, 32-bit PE binary with a single
// ".text" section covering RVAs [0x1000, 0x2000), optionally declaring a
// Load Config Table data directory (data directory index 10) pointing at
// \p LoadConfigRVA. This mirrors the layout that made Heroes3.exe (see
// #598) fail to open: a stale Load Config Table RVA left over by a linker
// that never actually emitted one.
static std::vector<uint8_t>
makeSyntheticPE32(std::optional<uint32_t> LoadConfigRVA) {
  constexpr uint32_t SizeOfHeaders = 0x200;
  constexpr uint32_t SizeOfRawData = 0x200;
  std::vector<uint8_t> File(SizeOfHeaders + SizeOfRawData, 0);

  auto put16 = [&](size_t Offset, uint16_t Value) {
    memcpy(File.data() + Offset, &Value, sizeof(Value));
  };
  auto put32 = [&](size_t Offset, uint32_t Value) {
    memcpy(File.data() + Offset, &Value, sizeof(Value));
  };

  // DOS header: just the "MZ" magic and e_lfanew pointing right after it.
  File[0] = 'M';
  File[1] = 'Z';
  constexpr uint32_t PEOffset = 64;
  put32(60, PEOffset);

  // PE signature.
  File[PEOffset] = 'P';
  File[PEOffset + 1] = 'E';
  File[PEOffset + 2] = 0;
  File[PEOffset + 3] = 0;

  // COFF file header.
  constexpr uint32_t COFFOffset = PEOffset + 4;
  constexpr uint16_t SizeOfOptionalHeader = 224; // pe32_header + 16 * 8
  put16(COFFOffset + 0, 0x14c);                  // Machine: i386
  put16(COFFOffset + 2, 1);                      // NumberOfSections
  put32(COFFOffset + 4, 0);                      // TimeDateStamp
  put32(COFFOffset + 8, 0);                      // PointerToSymbolTable
  put32(COFFOffset + 12, 0);                     // NumberOfSymbols
  put16(COFFOffset + 16, SizeOfOptionalHeader);
  put16(COFFOffset + 18, 0x0102); // EXECUTABLE_IMAGE | 32BIT_MACHINE

  // Optional header (pe32_header, 96 bytes).
  constexpr uint32_t OptOffset = COFFOffset + 20;
  put16(OptOffset + 0, 0x10b); // Magic: PE32
  put32(OptOffset + 16, 0x1000);        // AddressOfEntryPoint
  put32(OptOffset + 20, 0x1000);        // BaseOfCode
  put32(OptOffset + 24, 0x2000);        // BaseOfData
  put32(OptOffset + 28, 0x400000);      // ImageBase
  put32(OptOffset + 32, 0x1000);        // SectionAlignment
  put32(OptOffset + 36, 0x200);         // FileAlignment
  put32(OptOffset + 56, 0x2000);        // SizeOfImage
  put32(OptOffset + 60, SizeOfHeaders); // SizeOfHeaders
  put16(OptOffset + 68, 3);             // Subsystem
  put32(OptOffset + 92, 16);            // NumberOfRvaAndSize

  // Data directories (16 entries, 8 bytes each), right after pe32_header.
  constexpr uint32_t DataDirOffset = OptOffset + 96;
  if (LoadConfigRVA) {
    constexpr uint32_t LoadConfigIndex = 10;
    put32(DataDirOffset + LoadConfigIndex * 8, *LoadConfigRVA);
    put32(DataDirOffset + LoadConfigIndex * 8 + 4, 0x40);
  }

  // Section table: one ".text" section.
  constexpr uint32_t SectionOffset = DataDirOffset + 16 * 8;
  memcpy(File.data() + SectionOffset, ".text", 5);
  put32(SectionOffset + 8, 0x1000);          // VirtualSize
  put32(SectionOffset + 12, 0x1000);         // VirtualAddress
  put32(SectionOffset + 16, SizeOfRawData);  // SizeOfRawData
  put32(SectionOffset + 20, SizeOfHeaders);  // PointerToRawData
  put32(SectionOffset + 36, 0x60000020);     // CODE | EXECUTE | READ

  BOOST_TEST_REQUIRE(SectionOffset + 40 <= SizeOfHeaders);

  return File;
}

// Builds the corresponding minimal 64-bit PE binary. Keeping a separate
// helper makes the test exercise the PE32+ branch in the sanitizer rather than
// only checking that both header layouts happen to share the same offsets.
static std::vector<uint8_t>
makeSyntheticPE32Plus(std::optional<uint32_t> LoadConfigRVA) {
  constexpr uint32_t SizeOfHeaders = 0x200;
  constexpr uint32_t SizeOfRawData = 0x200;
  std::vector<uint8_t> File(SizeOfHeaders + SizeOfRawData, 0);

  auto put16 = [&](size_t Offset, uint16_t Value) {
    memcpy(File.data() + Offset, &Value, sizeof(Value));
  };
  auto put32 = [&](size_t Offset, uint32_t Value) {
    memcpy(File.data() + Offset, &Value, sizeof(Value));
  };
  auto put64 = [&](size_t Offset, uint64_t Value) {
    memcpy(File.data() + Offset, &Value, sizeof(Value));
  };

  File[0] = 'M';
  File[1] = 'Z';
  constexpr uint32_t PEOffset = 64;
  put32(60, PEOffset);

  File[PEOffset] = 'P';
  File[PEOffset + 1] = 'E';
  File[PEOffset + 2] = 0;
  File[PEOffset + 3] = 0;

  constexpr uint32_t COFFOffset = PEOffset + 4;
  constexpr uint16_t SizeOfOptionalHeader = 240; // pe32plus_header + 16 * 8
  put16(COFFOffset + 0, 0x8664);                 // Machine: AMD64
  put16(COFFOffset + 2, 1);                      // NumberOfSections
  put32(COFFOffset + 4, 0);                      // TimeDateStamp
  put32(COFFOffset + 8, 0);                      // PointerToSymbolTable
  put32(COFFOffset + 12, 0);                     // NumberOfSymbols
  put16(COFFOffset + 16, SizeOfOptionalHeader);
  put16(COFFOffset + 18, 0x0022); // EXECUTABLE_IMAGE | LARGE_ADDRESS_AWARE

  constexpr uint32_t OptOffset = COFFOffset + 20;
  put16(OptOffset + 0, 0x20b); // Magic: PE32+
  put32(OptOffset + 16, 0x1000);        // AddressOfEntryPoint
  put32(OptOffset + 20, 0x1000);        // BaseOfCode
  put64(OptOffset + 24, 0x140000000);   // ImageBase
  put32(OptOffset + 32, 0x1000);        // SectionAlignment
  put32(OptOffset + 36, 0x200);         // FileAlignment
  put32(OptOffset + 56, 0x2000);        // SizeOfImage
  put32(OptOffset + 60, SizeOfHeaders); // SizeOfHeaders
  put16(OptOffset + 68, 3);             // Subsystem
  put32(OptOffset + 108, 16);           // NumberOfRvaAndSize

  constexpr uint32_t DataDirOffset = OptOffset + 112;
  if (LoadConfigRVA) {
    constexpr uint32_t LoadConfigIndex = 10;
    put32(DataDirOffset + LoadConfigIndex * 8, *LoadConfigRVA);
    put32(DataDirOffset + LoadConfigIndex * 8 + 4, 0x40);
  }

  constexpr uint32_t SectionOffset = DataDirOffset + 16 * 8;
  memcpy(File.data() + SectionOffset, ".text", 5);
  put32(SectionOffset + 8, 0x1000);          // VirtualSize
  put32(SectionOffset + 12, 0x1000);         // VirtualAddress
  put32(SectionOffset + 16, SizeOfRawData);  // SizeOfRawData
  put32(SectionOffset + 20, SizeOfHeaders);  // PointerToRawData
  put32(SectionOffset + 36, 0x60000020);     // CODE | EXECUTE | READ

  BOOST_TEST_REQUIRE(SectionOffset + 40 <= SizeOfHeaders);

  return File;
}

static std::unique_ptr<MemoryBuffer>
toBuffer(const std::vector<uint8_t> &Bytes) {
  StringRef Data(reinterpret_cast<const char *>(Bytes.data()), Bytes.size());
  return MemoryBuffer::getMemBufferCopy(Data);
}

BOOST_AUTO_TEST_CASE(MalformedLoadConfigRvaIsPatchedAndBecomesOpenable) {
  // A Load Config Table RVA of 0x2a0 falls inside the header region, before
  // the ".text" section starts at 0x1000: llvm::object::createBinary must
  // reject it as-is.
  auto Buffer = toBuffer(makeSyntheticPE32(0x2a0));

  auto UnpatchedResult = createBinary(*Buffer);
  BOOST_TEST_REQUIRE(not UnpatchedResult);
  consumeError(UnpatchedResult.takeError());

  auto Patched = sanitizeCOFFLoadConfigDirectory(*Buffer);
  BOOST_TEST_REQUIRE(Patched != nullptr);

  auto PatchedResult = createBinary(*Patched);
  BOOST_TEST_REQUIRE(bool(PatchedResult));
}

BOOST_AUTO_TEST_CASE(WellFormedLoadConfigRvaIsLeftUntouched) {
  // A Load Config Table RVA inside the ".text" section is valid: the
  // sanitizer must be a no-op, and createBinary must already succeed.
  auto Buffer = toBuffer(makeSyntheticPE32(0x1000));

  BOOST_TEST(sanitizeCOFFLoadConfigDirectory(*Buffer) == nullptr);

  auto Result = createBinary(*Buffer);
  BOOST_TEST_REQUIRE(bool(Result));
}

BOOST_AUTO_TEST_CASE(MalformedPE32PlusLoadConfigRvaIsPatchedAndBecomesOpenable) {
  auto Buffer = toBuffer(makeSyntheticPE32Plus(0x2a0));

  auto UnpatchedResult = createBinary(*Buffer);
  BOOST_TEST_REQUIRE(not UnpatchedResult);
  consumeError(UnpatchedResult.takeError());

  auto Patched = sanitizeCOFFLoadConfigDirectory(*Buffer);
  BOOST_TEST_REQUIRE(Patched != nullptr);

  auto PatchedResult = createBinary(*Patched);
  BOOST_TEST_REQUIRE(bool(PatchedResult));
}

BOOST_AUTO_TEST_CASE(LoadConfigEntryOutsideDeclaredOptionalHeaderIsNotPatched) {
  auto Bytes = makeSyntheticPE32(0x2a0);

  // Keep the PE32 fixed header and NumberOfRvaAndSizes, but declare that the
  // optional header ends before data directory index 10. The bytes at that
  // offset must not be mistaken for a data directory and modified.
  constexpr uint32_t COFFOffset = 68;
  constexpr uint16_t TruncatedOptionalHeaderSize = 176;
  memcpy(Bytes.data() + COFFOffset + 16,
         &TruncatedOptionalHeaderSize,
         sizeof(TruncatedOptionalHeaderSize));

  auto Buffer = toBuffer(Bytes);
  BOOST_TEST(sanitizeCOFFLoadConfigDirectory(*Buffer) == nullptr);
}

BOOST_AUTO_TEST_CASE(MissingLoadConfigDirectoryIsLeftUntouched) {
  auto Buffer = toBuffer(makeSyntheticPE32(std::nullopt));

  BOOST_TEST(sanitizeCOFFLoadConfigDirectory(*Buffer) == nullptr);

  auto Result = createBinary(*Buffer);
  BOOST_TEST_REQUIRE(bool(Result));
}

BOOST_AUTO_TEST_CASE(NonPEBuffersAreLeftUntouched) {
  auto Empty = MemoryBuffer::getMemBufferCopy("");
  BOOST_TEST(sanitizeCOFFLoadConfigDirectory(*Empty) == nullptr);

  auto NotAnMZFile = MemoryBuffer::getMemBufferCopy("not a PE file at all");
  BOOST_TEST(sanitizeCOFFLoadConfigDirectory(*NotAnMZFile) == nullptr);

  // A buffer that only contains the "MZ" magic and an out-of-bounds
  // e_lfanew: must not crash, and must not be patched.
  std::vector<uint8_t> Tiny(4, 0);
  Tiny[0] = 'M';
  Tiny[1] = 'Z';
  auto TinyBuffer = toBuffer(Tiny);
  BOOST_TEST(sanitizeCOFFLoadConfigDirectory(*TinyBuffer) == nullptr);
}
