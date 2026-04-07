//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <array>
#include <cstdint>
#include <map>
#include <optional>
#include <string>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Object/COFF.h"
#include "llvm/Support/Endian.h"

#include "revng/Support/Generator.h"
#include "revng/Support/OverflowSafeInt.h"
#include "revng/Support/WindowsApiSetSchemaParser.h"

static Logger Log("api-set-schema-parser");

/// Template class to represent array of T with named fields
template<typename FieldsEnum, typename T>
class NamedArray {
public:
  using Fields = FieldsEnum;
  static constexpr size_t FieldCount = static_cast<size_t>(Fields::FieldCount);
  static constexpr auto Size = FieldCount * sizeof(T);

public:
  std::array<T, FieldCount> Storage{};

public:
  const T &operator[](Fields F) const {
    return Storage[static_cast<size_t>(F)];
  }
  T &operator[](Fields F) { return Storage[static_cast<size_t>(F)]; }

public:
  static std::optional<NamedArray> read(llvm::ArrayRef<uint8_t> Data,
                                        size_t Offset) {
    if (Offset >= Data.size())
      return std::nullopt;

    NamedArray Result;
    OverflowSafeInt<size_t> EndOffset = Offset;
    EndOffset += size(Result.Storage);
    if (not EndOffset or *EndOffset >= Data.size())
      return std::nullopt;

    for (size_t I = 0; I < FieldsEnum::FieldCount; ++I) {
      using namespace llvm::support;
      Result.Storage[I] = endian::read<T>(Data.data() + Offset
                                            + I * sizeof(uint32_t),
                                          llvm::support::little);
    }

    return Result;
  }
};

/// Reads an ASCII string encoded as UTF-16. Fails if it contains non-ASCII
/// characters.
static std::optional<std::string>
readUtf16LEASCIIString(llvm::ArrayRef<uint8_t> Data,
                       size_t Offset,
                       size_t LengthInBytes) {
  if (Offset > Data.size() or Offset + LengthInBytes > Data.size()
      or LengthInBytes % 2 != 0) {
    revng_log(Log, "Invalid coordinates for UTF-16 string");
    return std::nullopt;
  }

  std::string Result;
  for (size_t I = 0; I < LengthInBytes; I += 2) {
    uint16_t Char = Data[Offset + I] | (Data[Offset + I + 1] << 8);
    if (Char == 0)
      break;
    if (Char >= 128) {
      revng_log(Log, "Found non-ASCII character, bailing out");
      return std::nullopt;
    }
    Result += static_cast<char>(Char);
  }

  return Result;
}

class ApiSetSchema {
private:
  llvm::ArrayRef<uint8_t> Data;
  revng::WindowsLibraryMap Result;

private:
  ApiSetSchema(llvm::ArrayRef<uint8_t> Data) : Data(Data) {}

public:
  /// Main parsing function
  static std::optional<revng::WindowsLibraryMap>
  parse(llvm::ArrayRef<uint8_t> Data);

private:
  /// Parse ApiSet V2 (Windows 7)
  bool parseApiSetV2();

  /// Parse ApiSet V4 (Windows 8 and 8.1)
  bool parseApiSetV4();

  /// Parse ApiSet V6 (Windows 10 onwards)
  bool parseApiSetV6();

private:
  template<typename T>
  auto read(size_t Offset) const {
    return T::read(Data, Offset);
  }

  auto readString(size_t Offset, size_t LengthInBytes) const {
    return readUtf16LEASCIIString(Data, Offset, LengthInBytes);
  }

  template<typename T>
  cppcoro::generator<T> readArray(size_t Offset, size_t Count) {
    for (uint32_t I = 0; I < Count; ++I) {
      auto MaybeEntry = read<T>(Offset);
      if (not MaybeEntry.has_value()) {
        co_return;
      }
      auto &Entry = *MaybeEntry;
      co_yield Entry;
      Offset += T::Size;
    }
  }

private:
  void registerRedirect(llvm::StringRef Source, llvm::StringRef Destination) {
    revng_assert(not Source.empty());
    revng_assert(not Destination.empty());
    revng_log(Log, "Registering redirect: " << Source << " -> " << Destination);
    Result.Map.emplace(Source.lower(), Destination.lower());
  }
};

// ApiSet V2 Structure Definitions
namespace ApiSetV2 {

namespace ValueEntryRedirection {
enum Fields : size_t {
  NameOffset,
  NameLength,
  ValueOffset,
  ValueLength,
  FieldCount
};
using Class = NamedArray<Fields, uint32_t>;
} // namespace ValueEntryRedirection

namespace ValueEntry {
enum Fields : size_t {
  NumberOfRedirections,
  FieldCount
};
using Class = NamedArray<Fields, uint32_t>;
} // namespace ValueEntry

namespace NamespaceEntry {
enum Fields : size_t {
  NameOffset,
  NameLength,
  DataOffset,
  FieldCount
};
using Class = NamedArray<Fields, uint32_t>;
} // namespace NamespaceEntry

namespace Namespace {
enum Fields : size_t {
  Version,
  Count,
  FieldCount
};
using Class = NamedArray<Fields, uint32_t>;
} // namespace Namespace

} // namespace ApiSetV2

bool ApiSetSchema::parseApiSetV2() {
  using namespace ApiSetV2;

  auto MaybeNamespaceHeader = read<Namespace::Class>(0);

  if (not MaybeNamespaceHeader.has_value()) {
    revng_log(Log, "Couldn't read the namespace header");
    return false;
  }

  auto &NamespaceHeader = *MaybeNamespaceHeader;

  for (auto &Entry :
       readArray<NamespaceEntry::Class>(Namespace::Class::Size,
                                        NamespaceHeader[Namespace::Count])) {
    // Read the API set name
    auto ApiSetName = readString(Entry[NamespaceEntry::NameOffset],
                                 Entry[NamespaceEntry::NameLength]);
    if (not ApiSetName.has_value()) {
      revng_log(Log, "Nameless entry, skipping");
      continue;
    }

    revng_log(Log, "Considering " << *ApiSetName);
    LoggerIndent Indent(Log);

    // Read value entry
    size_t ValueEntryOffset = Entry[NamespaceEntry::DataOffset];
    auto MaybeValueEntryHeader = read<ValueEntry::Class>(ValueEntryOffset);
    if (not MaybeValueEntryHeader.has_value()) {
      revng_log(Log, "Couldn't read the entry, skipping");
      continue;
    }

    auto &ValueEntryHeader = *MaybeValueEntryHeader;

    // Read redirections
    auto StartOffset = ValueEntryOffset + ValueEntry::Class::Size;
    auto RedirectionCount = ValueEntryHeader[ValueEntry::NumberOfRedirections];
    for (auto &Redirection :
         readArray<ValueEntryRedirection::Class>(StartOffset,
                                                 RedirectionCount)) {

      auto TargetNameOffset = Redirection[ValueEntryRedirection::ValueOffset];
      auto TargetNameLength = Redirection[ValueEntryRedirection::ValueLength];

      if (TargetNameLength == 0) {
        revng_log(Log, "Found an entry of length 0, skipping");
        continue;
      }

      auto TargetName = readString(TargetNameOffset, TargetNameLength);

      if (not TargetName.has_value()) {
        revng_log(Log, "Couldn't read the string, skipping");
        continue;
      }

      registerRedirect(*ApiSetName, *TargetName);
    }
  }

  Result.PrefixLength = 4;

  return true;
}

// ApiSet V4 Structure Definitions
namespace ApiSetV4 {

namespace ValueEntryRedirection {
enum Fields : size_t {
  Flags,
  NameOffset,
  NameLength,
  ValueOffset,
  ValueLength,
  FieldCount
};
using Class = NamedArray<Fields, uint32_t>;
} // namespace ValueEntryRedirection

namespace ValueEntry {
enum Fields : size_t {
  Flags,
  NumberOfRedirections,
  FieldCount
};
using Class = NamedArray<Fields, uint32_t>;
} // namespace ValueEntry

namespace NamespaceEntry {
enum Fields : size_t {
  Flags,
  NameOffset,
  NameLength,
  AliasOffset,
  AliasLength,
  DataOffset,
  FieldCount
};
using Class = NamedArray<Fields, uint32_t>;
} // namespace NamespaceEntry

namespace Namespace {
enum Fields : size_t {
  Version,
  Size,
  Flags,
  Count,
  FieldCount
};
using Class = NamedArray<Fields, uint32_t>;
} // namespace Namespace

} // namespace ApiSetV4

bool ApiSetSchema::parseApiSetV4() {
  using namespace ApiSetV4;

  auto MaybeNamespaceHeader = read<Namespace::Class>(0);
  if (not MaybeNamespaceHeader.has_value()) {
    revng_log(Log, "Couldn't read the namespace header");
    return false;
  }
  auto &NamespaceHeader = *MaybeNamespaceHeader;

  for (auto &Entry :
       readArray<NamespaceEntry::Class>(Namespace::Class::Size,
                                        NamespaceHeader[Namespace::Count])) {

    // Read the API set name
    auto ApiSetName = readString(Entry[NamespaceEntry::NameOffset],
                                 Entry[NamespaceEntry::NameLength]);
    if (not ApiSetName.has_value()) {
      revng_log(Log, "Nameless entry, skipping");
      continue;
    }

    // Handle alias if present
    std::string FinalName = *ApiSetName;
    if (Entry[NamespaceEntry::AliasLength] > 0) {
      auto AliasName = readString(Entry[NamespaceEntry::AliasOffset],
                                  Entry[NamespaceEntry::AliasLength]);
      if (AliasName) {
        // An alias is a shorter version of the name, enabling wider matching
        // Example: ms-win-advapi32-auth-l1 for ms-win-advapi32-auth-l1-1-0
        FinalName = *AliasName;
      }
    }

    // Read value entry
    size_t ValueEntryOffset = Entry[NamespaceEntry::DataOffset];
    auto MaybeValueEntryHeader = read<ValueEntry::Class>(ValueEntryOffset);
    if (not MaybeValueEntryHeader.has_value()) {
      revng_log(Log, "Couldn't read the entry, skipping");
      continue;
    }
    auto &ValueEntryHeader = *MaybeValueEntryHeader;

    // Read redirections
    auto StartOffset = ValueEntryOffset + ValueEntry::Class::Size;
    auto RedirectionCount = ValueEntryHeader[ValueEntry::NumberOfRedirections];
    for (auto &Redirection :
         readArray<ValueEntryRedirection::Class>(StartOffset,
                                                 RedirectionCount)) {

      auto TargetNameOffset = Redirection[ValueEntryRedirection::ValueOffset];
      auto TargetNameLength = Redirection[ValueEntryRedirection::ValueLength];

      if (TargetNameLength == 0) {
        revng_log(Log, "Found an entry of length 0, skipping");
        continue;
      }

      auto TargetName = readString(TargetNameOffset, TargetNameLength);
      if (not TargetName.has_value()) {
        revng_log(Log, "Couldn't read the string, skipping");
        continue;
      }

      registerRedirect(*ApiSetName, *TargetName);
    }
  }

  Result.PrefixLength = 4;

  return true;
}

// ApiSet V6 Structure Definitions
namespace ApiSetV6 {

namespace HashEntry {
enum Fields : size_t {
  Hash,
  Index,
  FieldCount
};
using Class = NamedArray<Fields, uint32_t>;
} // namespace HashEntry

namespace NamespaceEntry {
enum Fields : size_t {
  Flags,
  NameOffset,
  NameLength,
  HashedLength,
  ValueOffset,
  ValueCount,
  FieldCount
};
using Class = NamedArray<Fields, uint32_t>;
} // namespace NamespaceEntry

namespace ValueEntry {
enum Fields : size_t {
  Flags,
  NameOffset,
  NameLength,
  ValueOffset,
  ValueLength,
  FieldCount
};
using Class = NamedArray<Fields, uint32_t>;
} // namespace ValueEntry

namespace Namespace {
enum Fields : size_t {
  Version,
  Size,
  Flags,
  Count,
  EntryOffset,
  HashOffset,
  HashFactor,
  FieldCount
};
using Class = NamedArray<Fields, uint32_t>;
} // namespace Namespace

} // namespace ApiSetV6

bool ApiSetSchema::parseApiSetV6() {
  using namespace ApiSetV6;

  auto MaybeNamespaceHeader = read<Namespace::Class>(0);
  if (not MaybeNamespaceHeader.has_value()) {
    return false;
  }
  auto &NamespaceHeader = *MaybeNamespaceHeader;

  if (Log.isEnabled()) {
    Log << "Namespace header:\n";
    Log << "  Version: " << NamespaceHeader[Namespace::Version] << "\n";
    Log << "  Size: " << NamespaceHeader[Namespace::Size] << "\n";
    Log << "  Flags: " << NamespaceHeader[Namespace::Flags] << "\n";
    Log << "  EntryOffset: " << NamespaceHeader[Namespace::EntryOffset] << "\n";
    Log << "  HashOffset: " << NamespaceHeader[Namespace::HashOffset] << "\n";
    Log << "  HashFactor: " << NamespaceHeader[Namespace::HashFactor] << "\n";
    Log << DoLog;
  }
  LoggerIndent Indent(Log);

  for (auto &Entry :
       readArray<NamespaceEntry::Class>(NamespaceHeader[Namespace::EntryOffset],
                                        NamespaceHeader[Namespace::Count])) {

    if (Log.isEnabled()) {
      Log << "Namespace entry:\n";
      Log << "  Flags: " << Entry[NamespaceEntry::Flags] << "\n";
      Log << "  NameOffset: " << Entry[NamespaceEntry::NameOffset] << "\n";
      Log << "  NameLength: " << Entry[NamespaceEntry::NameLength] << "\n";
      Log << "  HashedLength: " << Entry[NamespaceEntry::HashedLength] << "\n";
      Log << "  ValueOffset: " << Entry[NamespaceEntry::ValueOffset] << "\n";
      Log << "  ValueCount: " << Entry[NamespaceEntry::ValueCount] << "\n";
      Log << DoLog;
    }
    LoggerIndent Indent(Log);

    // Read the API set name
    auto ApiSetNamePrefix = readString(Entry[NamespaceEntry::NameOffset],
                                       Entry[NamespaceEntry::HashedLength]);
    if (not ApiSetNamePrefix.has_value()) {
      revng_log(Log, "Couldn't fetch entry's name prefix");
      return false;
    }

    if (Log.isEnabled()) {
      auto ApiSetName = readString(Entry[NamespaceEntry::NameOffset],
                                   Entry[NamespaceEntry::NameLength]);

      if (not ApiSetName.has_value()) {
        revng_log(Log, "Couldn't fetch entry's name");
        return false;
      }

      revng_log(Log, "Entry's name: " << ApiSetName.value());
      revng_log(Log, "Entry's prefix: " << ApiSetNamePrefix.value());
    }

    // Read value entries
    for (auto &ValueEntryItem :
         readArray<ValueEntry::Class>(Entry[NamespaceEntry::ValueOffset],
                                      Entry[NamespaceEntry::ValueCount])) {

      if (Log.isEnabled()) {
        Log << "Namespace value entry:\n";
        Log << "  Flags: " << ValueEntryItem[ValueEntry::Flags] << "\n";
        Log << "  NameOffset: " << ValueEntryItem[ValueEntry::NameOffset]
            << "\n";
        Log << "  NameLength: " << ValueEntryItem[ValueEntry::NameLength]
            << "\n";
        Log << "  ValueOffset: " << ValueEntryItem[ValueEntry::ValueOffset]
            << "\n";
        Log << "  ValueLength: " << ValueEntryItem[ValueEntry::ValueLength]
            << "\n";
        Log << DoLog;
      }

      auto TargetName = readString(ValueEntryItem[ValueEntry::ValueOffset],
                                   ValueEntryItem[ValueEntry::ValueLength]);
      if (not TargetName.has_value()) {
        revng_log(Log, "Couldn't fetch target's name");
        return false;
      }

      revng_log(Log, "Target: " << TargetName.value());

      if (TargetName->empty()) {
        revng_log(Log, "Empty target, skipping");
        continue;
      }

      registerRedirect(*ApiSetNamePrefix, *TargetName);
    }
  }

  return true;
}

std::optional<revng::WindowsLibraryMap>
ApiSetSchema::parse(llvm::ArrayRef<uint8_t> Data) {
  revng_log(Log, "Parsing API Schema Set section of size " << Data.size());
  LoggerIndent Indent(Log);

  if (Data.size() < 4) {
    revng_log(Log, "Section smaller than 4 bytes, bailing out.");
    return std::nullopt;
  }

  // Read version field
  using namespace llvm::support::endian;
  auto Version = read32(Data.data(), llvm::support::little);
  revng_log(Log, "Version " << Version);

  ApiSetSchema Result(Data);
  bool Success = false;

  switch (Version) {
  case 2:
    Success = Result.parseApiSetV2();
    break;
  case 4:
    Success = Result.parseApiSetV4();
    break;
  case 6:
    Success = Result.parseApiSetV6();
    break;
  }

  if (not Success)
    return std::nullopt;

  return std::move(Result.Result);
}

std::optional<revng::WindowsLibraryMap>
parseWindowsApiSetSchemaFromDLL(llvm::StringRef Path) {
  using namespace llvm::object;
  revng_log(Log, "Loading redirections from " << Path);

  auto MaybeObjectFile = createBinary(Path);

  if (not MaybeObjectFile) {
    std::string Message = llvm::toString(MaybeObjectFile.takeError());
    revng_log(Log, "Couldn't parse binary: " << Message);
    return {};
  }

  auto *PECOFF = llvm::dyn_cast<COFFObjectFile>(MaybeObjectFile->getBinary());
  if (PECOFF == nullptr) {
    revng_log(Log, "Not a PE/COFF binary");
    return {};
  }

  for (auto &Section : PECOFF->sections()) {
    auto MaybeName = Section.getName();
    if (not MaybeName) {
      std::string Message = llvm::toString(MaybeName.takeError());
      revng_log(Log, "Couldn't get the section name: " << Message);
      continue;
    }

    if (*MaybeName != ".apiset")
      continue;

    auto MaybeContents = Section.getContents();
    if (not MaybeContents) {
      std::string Message = llvm::toString(MaybeContents.takeError());
      revng_log(Log, "Couldn't get the section content: " << Message);
      continue;
    }

    if (MaybeContents->size() == 0) {
      revng_log(Log, "Section is empty");
      continue;
    }

    auto *Data = reinterpret_cast<const uint8_t *>(MaybeContents->data());
    llvm::ArrayRef<uint8_t> ApiSetSectionData(Data, MaybeContents->size());
    return ApiSetSchema::parse(ApiSetSectionData);
  }

  return {};
}
