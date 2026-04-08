#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <concepts>
#include <optional>

#include "llvm/ADT/StringExtras.h"
#include "llvm/BinaryFormat/ELF.h"
#include "llvm/Object/ELFObjectFile.h"
#include "llvm/Support/Endian.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Progress.h"

#include "revng/Model/Importer/Binary/BinaryDescriptor.h"
#include "revng/Model/Importer/DebugInfo/DwarfImporter.h"
#include "revng/Support/Configuration.h"
#include "revng/Support/Debug.h"
#include "revng/Support/Generator.h"
#include "revng/Support/ObjectFile.h"
#include "revng/Support/PathList.h"

#include "ImportDebugInfoHelper.h"

inline const std::string GlobalDebugDirectory = "/usr/lib/debug/";

inline cppcoro::generator<const llvm::sys::fs::directory_entry &>
listDirectory(const llvm::SmallString<16> &Path) {
  using namespace llvm::sys::fs;

  revng_log(DILogger, "Listing " << Path.str());

  std::error_code EC;
  auto DirectoryIt = directory_iterator(Path, EC);

  if (EC) {
    revng_log(DILogger,
              "Error enumerating " << Path.str() << ": " << EC.message());
  } else {

    for (const directory_iterator EndIt; DirectoryIt != EndIt;
         DirectoryIt.increment(EC)) {
      const auto &Entry = *DirectoryIt;
      co_yield Entry;
    }
  }
}

inline uint32_t gnuDebugLinkCRC32(llvm::ArrayRef<uint8_t> Data) {
  static const uint32_t Table[256] = {
    0x00000000, 0x77073096, 0xee0e612c, 0x990951ba, 0x076dc419, 0x706af48f,
    0xe963a535, 0x9e6495a3, 0x0edb8832, 0x79dcb8a4, 0xe0d5e91e, 0x97d2d988,
    0x09b64c2b, 0x7eb17cbd, 0xe7b82d07, 0x90bf1d91, 0x1db71064, 0x6ab020f2,
    0xf3b97148, 0x84be41de, 0x1adad47d, 0x6ddde4eb, 0xf4d4b551, 0x83d385c7,
    0x136c9856, 0x646ba8c0, 0xfd62f97a, 0x8a65c9ec, 0x14015c4f, 0x63066cd9,
    0xfa0f3d63, 0x8d080df5, 0x3b6e20c8, 0x4c69105e, 0xd56041e4, 0xa2677172,
    0x3c03e4d1, 0x4b04d447, 0xd20d85fd, 0xa50ab56b, 0x35b5a8fa, 0x42b2986c,
    0xdbbbc9d6, 0xacbcf940, 0x32d86ce3, 0x45df5c75, 0xdcd60dcf, 0xabd13d59,
    0x26d930ac, 0x51de003a, 0xc8d75180, 0xbfd06116, 0x21b4f4b5, 0x56b3c423,
    0xcfba9599, 0xb8bda50f, 0x2802b89e, 0x5f058808, 0xc60cd9b2, 0xb10be924,
    0x2f6f7c87, 0x58684c11, 0xc1611dab, 0xb6662d3d, 0x76dc4190, 0x01db7106,
    0x98d220bc, 0xefd5102a, 0x71b18589, 0x06b6b51f, 0x9fbfe4a5, 0xe8b8d433,
    0x7807c9a2, 0x0f00f934, 0x9609a88e, 0xe10e9818, 0x7f6a0dbb, 0x086d3d2d,
    0x91646c97, 0xe6635c01, 0x6b6b51f4, 0x1c6c6162, 0x856530d8, 0xf262004e,
    0x6c0695ed, 0x1b01a57b, 0x8208f4c1, 0xf50fc457, 0x65b0d9c6, 0x12b7e950,
    0x8bbeb8ea, 0xfcb9887c, 0x62dd1ddf, 0x15da2d49, 0x8cd37cf3, 0xfbd44c65,
    0x4db26158, 0x3ab551ce, 0xa3bc0074, 0xd4bb30e2, 0x4adfa541, 0x3dd895d7,
    0xa4d1c46d, 0xd3d6f4fb, 0x4369e96a, 0x346ed9fc, 0xad678846, 0xda60b8d0,
    0x44042d73, 0x33031de5, 0xaa0a4c5f, 0xdd0d7cc9, 0x5005713c, 0x270241aa,
    0xbe0b1010, 0xc90c2086, 0x5768b525, 0x206f85b3, 0xb966d409, 0xce61e49f,
    0x5edef90e, 0x29d9c998, 0xb0d09822, 0xc7d7a8b4, 0x59b33d17, 0x2eb40d81,
    0xb7bd5c3b, 0xc0ba6cad, 0xedb88320, 0x9abfb3b6, 0x03b6e20c, 0x74b1d29a,
    0xead54739, 0x9dd277af, 0x04db2615, 0x73dc1683, 0xe3630b12, 0x94643b84,
    0x0d6d6a3e, 0x7a6a5aa8, 0xe40ecf0b, 0x9309ff9d, 0x0a00ae27, 0x7d079eb1,
    0xf00f9344, 0x8708a3d2, 0x1e01f268, 0x6906c2fe, 0xf762575d, 0x806567cb,
    0x196c3671, 0x6e6b06e7, 0xfed41b76, 0x89d32be0, 0x10da7a5a, 0x67dd4acc,
    0xf9b9df6f, 0x8ebeeff9, 0x17b7be43, 0x60b08ed5, 0xd6d6a3e8, 0xa1d1937e,
    0x38d8c2c4, 0x4fdff252, 0xd1bb67f1, 0xa6bc5767, 0x3fb506dd, 0x48b2364b,
    0xd80d2bda, 0xaf0a1b4c, 0x36034af6, 0x41047a60, 0xdf60efc3, 0xa867df55,
    0x316e8eef, 0x4669be79, 0xcb61b38c, 0xbc66831a, 0x256fd2a0, 0x5268e236,
    0xcc0c7795, 0xbb0b4703, 0x220216b9, 0x5505262f, 0xc5ba3bbe, 0xb2bd0b28,
    0x2bb45a92, 0x5cb36a04, 0xc2d7ffa7, 0xb5d0cf31, 0x2cd99e8b, 0x5bdeae1d,
    0x9b64c2b0, 0xec63f226, 0x756aa39c, 0x026d930a, 0x9c0906a9, 0xeb0e363f,
    0x72076785, 0x05005713, 0x95bf4a82, 0xe2b87a14, 0x7bb12bae, 0x0cb61b38,
    0x92d28e9b, 0xe5d5be0d, 0x7cdcefb7, 0x0bdbdf21, 0x86d3d2d4, 0xf1d4e242,
    0x68ddb3f8, 0x1fda836e, 0x81be16cd, 0xf6b9265b, 0x6fb077e1, 0x18b74777,
    0x88085ae6, 0xff0f6a70, 0x66063bca, 0x11010b5c, 0x8f659eff, 0xf862ae69,
    0x616bffd3, 0x166ccf45, 0xa00ae278, 0xd70dd2ee, 0x4e048354, 0x3903b3c2,
    0xa7672661, 0xd06016f7, 0x4969474d, 0x3e6e77db, 0xaed16a4a, 0xd9d65adc,
    0x40df0b66, 0x37d83bf0, 0xa9bcae53, 0xdebb9ec5, 0x47b2cf7f, 0x30b5ffe9,
    0xbdbdf21c, 0xcabac28a, 0x53b39330, 0x24b4a3a6, 0xbad03605, 0xcdd70693,
    0x54de5729, 0x23d967bf, 0xb3667a2e, 0xc4614ab8, 0x5d681b02, 0x2a6f2b94,
    0xb40bbe37, 0xc30c8ea1, 0x5a05df1b, 0x2d02ef8d
  };

  uint32_t Result = 0;
  Result = ~Result & 0xffffffff;

  for (uint8_t Byte : Data)
    Result = Table[(Result ^ Byte) & 0xff] ^ (Result >> 8);

  return ~Result & 0xffffffff;
}

template<IsELFObjectFile T>
class DwarfDebugInfoFinder {
public:
  const revng::RootEntry &Root;
  const ELFBinary &Binary;
  const T &ObjectFile;

private:
  DwarfDebugInfoFinder(const revng::RootEntry &Root,
                       const ELFBinary &Binary,
                       const T &ObjectFile) :
    Root(Root), Binary(Binary), ObjectFile(ObjectFile) {}

public:
  static std::string find(const revng::RootEntry &Root,
                          const ELFBinary &Binary,
                          const T &ObjectFile) {
    DwarfDebugInfoFinder Finder(Root, Binary, ObjectFile);
    return Finder.find();
  }

private:
  std::string find() {
    using namespace llvm;

    revng_log(DILogger, "Looking for separate debug info");
    LoggerIndent Indent(DILogger);

    llvm::Task T1(3, "Looking for separate debug info");

    std::string SeparateDebugInfoPath;

    // Try to identify the separate debug info file locally
    revng_log(DILogger, "Searching locally");
    T1.advance("Searching locally", true);
    LoggerIndent Indent2(DILogger);
    SeparateDebugInfoPath = getSeparateDebugInfo();

    if (SeparateDebugInfoPath.empty()) {
      // We couldn't find locally, run fetch-debuginfo and try again
      revng_log(DILogger, "Fetching debug info remotely");
      T1.advance("Fetching debug info remotely", true);
      LoggerIndent Indent(DILogger);
      int ExitCode = runFetchDebugInfo(Binary.FullPathForExternalTools,
                                       DILogger.isEnabled());
      if (ExitCode == 0) {
        revng_log(DILogger, "Searching locally again");
        T1.advance("Searching locally again", true);
        LoggerIndent Indent(DILogger);
        SeparateDebugInfoPath = getSeparateDebugInfo();
      } else {
        revng_log(DILogger,
                  "Failed to find debug info with `revng model "
                  "fetch-debuginfo`.");
      }
    }

    return SeparateDebugInfoPath;
  }

  std::string getSeparateDebugInfo() {
    // First, use build-id in all of its forms, i.e., .note.gnu.build-id and
    // .gnu_debuglink.
    // Then consider .gnu_debugaltlink and manual look up by .debug suffix
    // (e.g., bash.debug).

    std::string Path = processBuildID();

    if (Path.size() == 0) {
      Path = processGnuDebugLink();
    }

    if (Path.size() == 0) {
      Path = processGnuDebugAltLink();
    }

    if (Path.size() == 0) {
      Path = processDotDebugFile();
    }

    if (Path.size() == 0) {
      revng_log(DILogger, "Not found");
    } else {
      revng_log(DILogger, "Found: " << Path);
    }

    return Path;
  }

  std::string processBuildID() {
    revng_log(DILogger, "Processing build ID");
    LoggerIndent Indent(DILogger);

    std::string BuildID = getBuildID();
    if (BuildID.size() == 0) {
      revng_log(DILogger, "Can't parse build-id.");
      return "";
    }

    revng_log(DILogger, "Build-id: " << BuildID);

    llvm::SmallString<128> ResultPath;

    {
      // First two chars of build-id forms the debug info file directory.
      auto DebugDir = BuildID.substr(0, 2);

      // The rest of build-id forms the debug info file name.
      auto DebugFile = BuildID.substr(2);
      auto DebugFileWithExtension = DebugFile.append(".debug");
      llvm::sys::path::append(ResultPath,
                              GlobalDebugDirectory,
                              ".build-id/",
                              DebugDir,
                              DebugFileWithExtension);

      if (auto MaybeFullPath = Root.getExistingPath(ResultPath.str())) {
        revng_log(DILogger, "Debug info found in " << *MaybeFullPath);
        return *MaybeFullPath;
      }
    }

    // Try in revng's debug symbols cache
    ResultPath.clear();
    std::string CacheDir = getCacheDirectory();
    ResultPath = joinPath(CacheDir, "debug-symbols", "elf", BuildID, "debug");

    if (fileExists(ResultPath.str())) {
      revng_log(DILogger, "Debug info found in cache: " << ResultPath.str());
      return ResultPath.str().str();
    }

    revng_log(DILogger, "Processing build-id did not lead to an existing file");
    return "";
  }

  llvm::ArrayRef<uint8_t> findELFNote(llvm::StringRef Name, uint32_t Type) {
    const auto &ELF = ObjectFile.getELFFile();

    auto MaybeProgramHeaders = ELF.program_headers();
    if (not MaybeProgramHeaders) {
      std::string Message = llvm::toString(MaybeProgramHeaders.takeError());
      revng_log(DILogger, "Couldn't parse program headers: " << Message);
      return {};
    }

    for (const auto &ProgramHeader : *MaybeProgramHeaders) {
      if (ProgramHeader.p_type != llvm::ELF::PT_NOTE)
        continue;

      llvm::Error NotesError = llvm::Error::success();
      for (const auto &ELFNote : ELF.notes(ProgramHeader, NotesError)) {
        if (ELFNote.getName() == Name and ELFNote.getType() == Type) {
          return ELFNote.getDesc();
        }
      }

      if (NotesError) {
        std::string Message = llvm::toString(std::move(NotesError));
        revng_log(DILogger, "Failed to parse notes: " << Message);
        return {};
      }
    }

    return {};
  }

  std::string getBuildID() {
    using namespace llvm;

    const auto &ELF = ObjectFile.getELFFile();

    // This is the correct way to identify .note.gnu.build-id
    ArrayRef<uint8_t> Contents = findELFNote("GNU", ELF::NT_GNU_BUILD_ID);
    if (Contents.size() <= 2) {
      // We want at least two bytes
      revng_log(DILogger, "Empty build-id found, skipping");
      return "";
    }

    std::string StringForBytes;
    {
      // Convert to hex
      raw_string_ostream OutputStream(StringForBytes);
      for (uint8_t Byte : Contents)
        OutputStream << format_hex_no_prefix(Byte, 2);
    }

    return StringForBytes;
  }

  std::string processGnuDebugLink() {
    revng_log(DILogger, "Processing .gnu_debuglink");
    LoggerIndent Indent(DILogger);

    auto [Path, Rest] = getStringFromSection(".gnu_debuglink");
    // .gnu_debuglink contains something like
    // d12d07f91f419ba28f2ab11d96355a54b0c2c2.debug or libc-2.31.so (which
    // could resolve to the input binary itself), followed by 0 to 3 bytes to
    // reach 4-bytes alignment and then a 4-bytes CRC.

    // Bail out if it's a path, as opposed to just a file name
    if (Path.contains("/")) {
      revng_log(DILogger,
                ".gnu_debuglink contains /, unexpected, bailing out.");
      return "";
    }

    if (Path.size() == 0)
      return {};

    // Collect the CRC
    if (Rest.size() < 4) {
      revng_log(DILogger, "No CRC found in .gnu_debuglink");
      return "";
    }

    using namespace llvm::object;
    bool IsLittleEndian = true;
    llvm::support::endianness Endianness = llvm::support::little;
    if constexpr (std::is_same_v<T, ELF32BEObjectFile>
                  or std::is_same_v<T, ELF64BEObjectFile>) {
      Endianness = llvm::support::big;
    }

    auto CRC = llvm::support::endian::read<uint32_t>(Rest.end() - 4,
                                                     Endianness);
    return findByName(Path, CRC);
  }

  std::string processGnuDebugAltLink() {
    revng_log(DILogger, "Processing .gnu_debugaltlink");
    LoggerIndent Indent(DILogger);

    // Look for the absolute path
    auto [Path, _] = getStringFromSection(".gnu_debugaltlink");

    if (Path.size() == 0)
      return {};

    if (auto MaybeFullPath = Root.getExistingPath(Path)) {
      return *MaybeFullPath;
    }

    // Not found: extract the file name and look in canonical places
    auto FileName = llvm::sys::path::filename(Path);

    return findByName(FileName, std::nullopt);
  }

  std::string processDotDebugFile() {
    revng_log(DILogger, "Processing the .debug file");
    LoggerIndent Indent(DILogger);

    auto FileName = llvm::sys::path::filename(Binary.canonicalPath());

    if (FileName.empty())
      return "";

    return findByName((llvm::Twine(FileName) + ".debug").str(), std::nullopt);
  }

private:
  llvm::ArrayRef<uint8_t> getSectionsContents(llvm::StringRef Name) {
    auto ELF = ObjectFile.getELFFile();
    auto MaybeSections = ELF.sections();
    if (auto Error = MaybeSections.takeError()) {
      // TODO: emit a diagnostic message for the user.
      revng_log(DILogger, "Failed to get sections: " << Error);
      consumeError(std::move(Error));
      return {};
    }

    for (const auto &Section : *MaybeSections) {
      auto MaybeName = expectedToOptional(ELF.getSectionName(Section));
      if (MaybeName and *MaybeName == Name) {
        auto MaybeContents = expectedToOptional(ELF
                                                  .getSectionContents(Section));
        if (MaybeContents)
          return *MaybeContents;
      }
    }

    return {};
  }

  std::pair<llvm::StringRef, llvm::ArrayRef<uint8_t>>
  getStringFromSection(llvm::StringRef Name) {
    using namespace llvm;

    auto ELF = ObjectFile.getELFFile();
    ArrayRef<uint8_t> SectionData = getSectionsContents(Name);
    if (SectionData.size() == 0)
      return {};

    for (auto [Index, Byte] : llvm::enumerate(SectionData)) {
      if (Byte == 0) {
        const char *Start = reinterpret_cast<const char *>(SectionData.data());
        return { StringRef(Start, Index), SectionData.slice(Index + 1) };
      }
    }

    return { {}, SectionData };
  }

  std::optional<std::string> getPathWithCRC(llvm::StringRef RootRelativePath,
                                            std::optional<uint32_t> CRC) {
    auto MaybePath = Root.getExistingPath(RootRelativePath);
    if (not MaybePath)
      return std::nullopt;

    if (CRC.has_value()) {
      revng_log(DILogger,
                "Checking if CRC is 0x" << llvm::utohexstr(*CRC, true));
      LoggerIndent Indent(DILogger);

      auto MaybeFile = llvm::MemoryBuffer::getFile(*MaybePath, false, false);
      if (std::error_code ErrorCode = MaybeFile.getError()) {
        revng_log(DILogger,
                  "Error opening the file to compute the CRC: " << ErrorCode);
        return std::nullopt;
      }

      auto ContentString = MaybeFile->get()->getBuffer();
      auto *Pointer = reinterpret_cast<const uint8_t *>(ContentString.data());
      auto ContentArray = llvm::ArrayRef<uint8_t>(Pointer,
                                                  ContentString.size());

      if (gnuDebugLinkCRC32(ContentArray) != *CRC) {
        revng_log(DILogger, "CRC mismatch");
        return std::nullopt;
      }
    }

    return MaybePath;
  }

  std::string findByName(llvm::StringRef Name, std::optional<uint32_t> CRC) {
    revng_assert(not Name.empty());

    using namespace llvm::sys::path;
    using namespace llvm::sys::fs;
    auto BinaryCanonicalPath = Binary.canonicalPath();
    bool HasAbsoluteCanonicalPath = is_absolute(BinaryCanonicalPath)
                                    and has_parent_path(BinaryCanonicalPath);

    revng_log(DILogger,
              "Looking for \"" << Name << "\" for \"" << BinaryCanonicalPath
                               << "\"");
    LoggerIndent Indent(DILogger);

    // If the binary canonical path is `/usr/bin/ls`, we look for:
    //
    // * /usr/bin/$DEBUG_FILE_NAME
    // * /usr/bin/.debug/$DEBUG_FILE_NAME
    // * /usr/lib/debug/usr/bin/$DEBUG_FILE_NAME
    // * /usr/lib/debug/.build-id/*/$DEBUG_FILE_NAME
    //
    // Typically $DEBUG_FILE_NAME is ls.debug or
    // d12d07f91f419ba28f2ab11d96355a54b0c2c2.debug
    //
    // Note: in theory we could look in $XDG_CACHE_DIR/revng/debug-symbols/elf/
    // as well but we store files as $BUILD_ID/debug. Also, $BUILD_ID is the
    // full build-id, while the name we get here is stripped of the first byte
    // (two hex-digits).

    llvm::SmallString<128> ResultPath;

    if (HasAbsoluteCanonicalPath) {
      // Try in the same directory
      append(ResultPath, parent_path(BinaryCanonicalPath), Name);

      if (auto MaybeFullPath = getPathWithCRC(ResultPath.str(), CRC)) {
        return *MaybeFullPath;
      }

      // Try in .debug/ directory.
      ResultPath.clear();
      append(ResultPath, parent_path(BinaryCanonicalPath), ".debug/", Name);

      if (auto MaybeFullPath = getPathWithCRC(ResultPath.str(), CRC)) {
        return *MaybeFullPath;
      }

      // Try `/usr/lib/debug/usr/bin/ls.debug`-like path.
      ResultPath.clear();
      append(ResultPath,
             GlobalDebugDirectory,
             parent_path(BinaryCanonicalPath),
             Name);

      if (auto MaybeFullPath = getPathWithCRC(ResultPath.str(), CRC)) {
        return *MaybeFullPath;
      }
    }

    // Look in /usr/lib/debug/.build-id/*/
    // TODO: enumerating all the directories in .build-id is not very nice.
    llvm::SmallString<16> BasePath;
    append(BasePath, Root.Path, GlobalDebugDirectory, ".build-id");

    for (const auto &Entry : listDirectory(BasePath)) {
      if (Entry.type() != file_type::directory_file)
        continue;

      auto DirectoryName = filename(Entry.path());
      ResultPath.clear();
      append(ResultPath,
             GlobalDebugDirectory,
             ".build-id",
             DirectoryName,
             Name);

      if (auto MaybeFullPath = getPathWithCRC(ResultPath.str(), CRC)) {
        return *MaybeFullPath;
      }
    }

    return {};
  }

  bool fileExists(const llvm::Twine &Path) const {
    using namespace llvm::sys;
    revng_assert(not Path.str().empty());
    bool Result = fs::exists(Path);
    if (not Result)
      revng_log(DILogger, "No file at the following path: " << Path.str());
    return Result;
  }
};
