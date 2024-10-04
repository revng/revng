/// \file LDDTree.cpp
/// Implementation of lddtree like API.

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

extern "C" {
#include "glob.h"
}
#include <map>
#include <sstream>
#include <vector>

#include "llvm/ADT/StringRef.h"
#include "llvm/Object/Binary.h"
#include "llvm/Object/ELFObjectFile.h"
#include "llvm/Object/ObjectFile.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Process.h"

#include "revng/ADT/RecursiveCoroutine.h"
#include "revng/ADT/STLExtras.h"
#include "revng/Support/Debug.h"
#include "revng/Support/Generator.h"
#include "revng/Support/LDDTree.h"
#include "revng/Support/OverflowSafeInt.h"

static Logger<> Log("lddtree");

using namespace llvm;

constexpr unsigned MaxIncludeDepth = 5;

/// \see ldconfig.c from glibc
class LdSoConfParser {
private:
  SmallVectorImpl<std::string> &SearchPaths;
  std::set<std::string> VisitedFiles;

public:
  LdSoConfParser(SmallVectorImpl<std::string> &SearchPaths) :
    SearchPaths(SearchPaths) {}

public:
  void parse() {
    std::set<std::string> VisitedFiles;
    // TODO: add support for prefix?
    parseImpl("/etc/ld.so.conf", 0);
  }

private:
  void parseImpl(StringRef Path, unsigned Depth) {
    revng_log(Log, "Parsing " << Path);

    if (Depth > MaxIncludeDepth) {
      revng_log(Log,
                "More than " << MaxIncludeDepth
                             << " nested includes in ld.so.conf");
      return;
    }

    bool IsNew = VisitedFiles.insert(Path.str()).second;
    if (not IsNew) {
      revng_log(Log, "We already visited " << Path.str() << ". Ignoring.");
      return;
    }

    using namespace llvm::sys;
    StringRef Directory = path::parent_path(Path);

    auto MaybeBuffer = llvm::MemoryBuffer::getFile(Path);
    if (not MaybeBuffer) {
      revng_log(Log, "Can't open " << Path);
      return;
    }

    StringRef Data = MaybeBuffer->get()->getBuffer();

    // Split in lines
    SmallVector<StringRef, 8> Lines;
    Data.split(Lines, "\n");

    for (StringRef Line : Lines) {
      // Remove comments
      Line = Line.split("#").first;

      // Strip white spaces
      Line = Line.trim();

      // Ignore empty lines
      if (Line.size() == 0)
        continue;

      if (Line.consume_front("include ")) {
        Line = Line.trim();

        SmallString<32> GlobExpression = makeAbsolute(Line, Directory);
        for (StringRef File : glob(GlobExpression))
          parseImpl(File, Depth + 1);

      } else if (Line.consume_front("hwcap ")) {

        revng_log(Log, "Ignoring hwcap line");

      } else {

        // Add the path
        SmallString<32> SearchPath = makeAbsolute(Line, Directory);
        revng_log(Log, "Registering " << SearchPath.str().str());
        SearchPaths.push_back(SearchPath.str().str());
      }
    }
  }

private:
  static SmallString<32> makeAbsolute(StringRef Path,
                                      StringRef CurrentDirectory) {
    SmallString<32> Result;

    if (Path.starts_with("/")) {
      Result.append(Path);
    } else {
      Result = CurrentDirectory;
      llvm::sys::path::append(Result, Path);
    }

    return Result;
  }

  static cppcoro::generator<StringRef> glob(StringRef Pattern) {
    glob64_t GlobResults;
    int Result = glob64(Pattern.str().data(), 0, NULL, &GlobResults);

    switch (Result) {
    case 0:
      for (size_t I = 0; I < GlobResults.gl_pathc; ++I)
        co_yield StringRef(GlobResults.gl_pathv[I]);
      globfree64(&GlobResults);
      break;

    case GLOB_NOMATCH:
      break;

    case GLOB_NOSPACE:
    case GLOB_ABORTED:
      revng_log(Log, "Cannot read directory");
      break;

    default:
      revng_abort();
      break;
    }
  }
};

/// \see man ld.so
static std::optional<std::string>
findLibrary(StringRef ToImport,
            StringRef ImporterPath,
            bool Is64,
            bool NoDefault,
            uint16_t EMachine,
            std::optional<StringRef> RPath,
            std::optional<StringRef> RunPath) {
  revng_log(Log, "Looking for " << ToImport);
  LoggerIndent<> Indent(Log);

  SmallVector<std::string, 16> SearchPaths;

  // Process DT_RPATH
  if (RPath and RPath->size())
    for (StringRef Path : split(*RPath, ":"))
      SearchPaths.push_back(Path.str());

  // Process the `LD_LIBRARY_PATH`
  if (auto MaybeLibraryPath = llvm::sys::Process::GetEnv("LD_LIBRARY_PATH"))
    for (StringRef Path : split(*MaybeLibraryPath, ":"))
      SearchPaths.push_back(Path.str());

  // Process DT_RUNPATH
  std::string Origin = llvm::sys::path::parent_path(ImporterPath).str();
  std::string LibName = Is64 ? "lib64" : "lib";
  if (RunPath and RunPath->size()) {
    for (StringRef Path : split(*RunPath, ":")) {
      std::string PathString = Path.str();
      replaceAll(PathString, "$ORIGIN", Origin);
      replaceAll(PathString, "${ORIGIN}", Origin);
      replaceAll(PathString, "$LIB", LibName);
      replaceAll(PathString, "${LIB}", LibName);
      // TODO: handle PLATFORM
      SearchPaths.push_back(PathString);
    }
  }

  if (not NoDefault) {
    LdSoConfParser(SearchPaths).parse();
    SearchPaths.push_back("/" + LibName);
    SearchPaths.push_back("/usr/" + LibName);
  }

  if (Log.isEnabled()) {
    Log << "List of search paths:\n";
    for (const std::string &SearchPath : SearchPaths)
      Log << "  " << SearchPath << "\n";
    Log << DoLog;
  }

  for (std::string SearchPath : SearchPaths) {
    SmallString<128> Candidate;
    sys::path::append(Candidate, SearchPath, ToImport);

    if (not sys::fs::exists(Candidate)) {
      revng_log(Log, Candidate.str() << " does not exist");
      continue;
    }

    // Parse the binary
    auto BinaryOrErr = object::createBinary(Candidate);
    if (not BinaryOrErr) {
      revng_log(Log,
                "Can't create binary: " << toString(BinaryOrErr.takeError()));
      llvm::consumeError(BinaryOrErr.takeError());
      continue;
    }

    // Ensure it's an ELF
    using namespace object;
    auto *Elf = dyn_cast<ELFObjectFileBase>(BinaryOrErr->getBinary());
    if (Elf == nullptr) {
      revng_log(Log, "Found " << Candidate.str() << " but it's not an ELF.");
      continue;
    }

    // Ensure it's the right machine
    if (Elf->getEMachine() != EMachine) {
      revng_log(Log,
                "Found " << Candidate.str()
                         << " but it has the wrong e_machine: "
                         << Elf->getEMachine() << " (expected " << EMachine
                         << ").");
      continue;
    }

    revng_log(Log, "Found: " << Candidate.str());

    return { Candidate.str().str() };
  }

  revng_log(Log, ToImport << " not found");
  return std::nullopt;
}

template<class ELFT>
std::optional<StringRef>
getDynamicString(const llvm::object::ELFFile<ELFT> &TheELF,
                 StringRef DynamicStringTable,
                 uint64_t Value) {
  if (DynamicStringTable.empty() && !DynamicStringTable.data())
    return std::nullopt;
  uint64_t FileSize = TheELF.getBufSize();
  uint64_t Offset = reinterpret_cast<const uint8_t *>(DynamicStringTable.data())
                    - TheELF.base();
  if (DynamicStringTable.size() > FileSize - Offset)
    return std::nullopt;
  if (Value >= DynamicStringTable.size())
    return std::nullopt;
  if (DynamicStringTable.back() != '\0')
    return std::nullopt;
  return DynamicStringTable.data() + Value;
}

template<class ELFT>
static std::optional<StringRef>
getDynamicStringTable(const ELFT &ELFObjectFile,
                      typename ELFT::Elf_Dyn_Range &DynamicEntries) {
  // Find address and size of .dynstr using dynamic entries
  uint64_t DynstrAddress = 0;
  uint64_t DynstrSize = 0;
  for (const auto &DynamicTag : DynamicEntries) {
    auto TheTag = DynamicTag.getTag();
    auto TheVal = DynamicTag.getVal();
    if (TheTag == llvm::ELF::DT_STRTAB) {
      DynstrAddress = TheVal;
    } else if (TheTag == llvm::ELF::DT_STRSZ) {
      DynstrSize = TheVal;
    }
  }

  // Compute end address
  auto MaybeEndAddress = OverflowSafeInt(DynstrAddress) + DynstrSize;
  if (DynstrAddress == 0 or DynstrSize == 0 or not MaybeEndAddress)
    return {};
  uint64_t EndAddress = *MaybeEndAddress;

  //
  // Convert address to offset
  //

  // Collect program headers
  auto ProgramHeadersOrError = ELFObjectFile.getELFFile().program_headers();
  if (not ProgramHeadersOrError) {
    revng_log(Log, "No valid program headers available");
    llvm::consumeError(ProgramHeadersOrError.takeError());
    return {};
  }

  // Find the correct program header
  using Elf_Phdr = ELFT::Elf_Phdr;
  for (const Elf_Phdr &Phdr : *ProgramHeadersOrError) {
    if (Phdr.p_type != llvm::ELF::PT_LOAD)
      continue;

    uint64_t StartAddress = Phdr.p_vaddr;
    uint64_t FileSize = Phdr.p_filesz;
    auto SegmentEndAddress = OverflowSafeInt(StartAddress) + FileSize;
    if (SegmentEndAddress and DynstrAddress >= Phdr.p_vaddr
        and EndAddress <= *SegmentEndAddress) {
      uint64_t SegmentStartOffset = Phdr.p_offset;
      auto MaybeSegmentEndOffset = OverflowSafeInt(SegmentStartOffset)
                                   + FileSize;
      auto MaybeDynstrOffset = OverflowSafeInt(DynstrAddress) - StartAddress;
      auto MaybeDynstrEnd = MaybeDynstrOffset + DynstrSize;

      StringRef RawData = ELFObjectFile.getData();
      if (MaybeSegmentEndOffset and MaybeDynstrOffset and MaybeDynstrEnd
          and SegmentStartOffset <= *MaybeSegmentEndOffset
          and SegmentStartOffset <= RawData.size()
          and *MaybeSegmentEndOffset <= RawData.size()) {
        return RawData.slice(SegmentStartOffset, *MaybeSegmentEndOffset)
          .slice(*MaybeDynstrOffset, *MaybeDynstrEnd);
      }
    }
  }

  return {};
}

template<class ELFT>
void lddtreeResolve(LDDTree &Dependencies,
                    StringRef FileName,
                    const ELFT &ELFObjectFile) {
  const auto &TheELF = ELFObjectFile.getELFFile();

  auto MaybeDynamicEntries = TheELF.dynamicEntries();

  if (auto Error = MaybeDynamicEntries.takeError()) {
    revng_log(Log, "Cannot access dynamic entries: " << Error);
    consumeError(std::move(Error));
    return;
  }

  using Elf_Dyn_Range = ELFT::Elf_Dyn_Range;
  Elf_Dyn_Range DynamicEntries = *MaybeDynamicEntries;

  std::optional<StringRef> RunPath;
  std::optional<StringRef> RPath;
  bool NoDefault = false;

  // Look for .dynstr
  StringRef DynamicStringTable;
  if (auto MaybeDynamicStringTable = getDynamicStringTable(ELFObjectFile,
                                                           DynamicEntries))
    DynamicStringTable = *MaybeDynamicStringTable;

  if (DynamicStringTable.empty()) {
    revng_log(Log, "Cannot find .dynstr");
    return;
  }

  // Look for DT_RPATH and DT_RUNPATH
  using Elf_Dyn = ELFT::Elf_Dyn;
  for (const Elf_Dyn &DynamicTag : DynamicEntries) {
    auto TheTag = DynamicTag.getTag();
    auto TheVal = DynamicTag.getVal();
    if (TheTag == llvm::ELF::DT_RUNPATH) {
      RunPath = getDynamicString(TheELF, DynamicStringTable, TheVal);
    } else if (TheTag == llvm::ELF::DT_RPATH) {
      RPath = getDynamicString(TheELF, DynamicStringTable, TheVal);
    } else if (TheTag == llvm::ELF::DT_FLAGS_1) {
      NoDefault = (TheVal & llvm::ELF::DF_1_NODEFLIB) != 0;
    }
  }

  bool Is64 = (std::is_same_v<ELFT, object::ELF64LEObjectFile>
               or std::is_same_v<ELFT, object::ELF64BEObjectFile>);

  for (const Elf_Dyn &DynamicTag : DynamicEntries) {
    auto TheTag = DynamicTag.getTag();
    auto TheVal = DynamicTag.getVal();
    if (TheTag == llvm::ELF::DT_NEEDED) {
      auto LibName = getDynamicString(TheELF, DynamicStringTable, TheVal);
      if (!LibName) {
        revng_log(Log, "Unable to parse needed library name");
        continue;
      }

      if (auto LocOfLib = findLibrary(*LibName,
                                      FileName,
                                      Is64,
                                      NoDefault,
                                      TheELF.getHeader().e_machine,
                                      RPath,
                                      RunPath))
        Dependencies[FileName.str()].push_back(*LocOfLib);
    }
  }
}

static RecursiveCoroutine<void> lddtreeHelper(LDDTree &Dependencies,
                                              const std::string &Path,
                                              unsigned CurrentLevel,
                                              unsigned DepthLevel) {
  revng_log(Log, "lddtree for " << Path << "\n");
  LoggerIndent<> Ident(Log);

  using namespace object;
  auto BinaryOrErr = createBinary(Path);
  if (not BinaryOrErr) {
    revng_log(Log,
              "Can't create binary: " << toString(BinaryOrErr.takeError()));
    llvm::consumeError(BinaryOrErr.takeError());
    rc_return;
  }

  auto *Binary = BinaryOrErr->getBinary();
  if (auto *ELFObjectFile = dyn_cast<ELF32LEObjectFile>(Binary))
    lddtreeResolve(Dependencies, Path, *ELFObjectFile);
  else if (auto *ELFObjectFile = dyn_cast<ELF32BEObjectFile>(Binary))
    lddtreeResolve(Dependencies, Path, *ELFObjectFile);
  else if (auto *ELFObjectFile = dyn_cast<ELF64LEObjectFile>(Binary))
    lddtreeResolve(Dependencies, Path, *ELFObjectFile);
  else if (auto *ELFObjectFile = dyn_cast<ELF64BEObjectFile>(Binary))
    lddtreeResolve(Dependencies, Path, *ELFObjectFile);
  else
    revng_log(Log, "Not an ELF.");

  if (CurrentLevel == DepthLevel)
    rc_return;

  ++CurrentLevel;
  for (auto &I : Dependencies) {
    revng_log(Log, "Dependencies for " << I.first << ":\n");
    for (auto &J : I.second)
      if (!Dependencies.contains(J))
        rc_recur lddtreeHelper(Dependencies, J, CurrentLevel, DepthLevel);
  }
}

void lddtree(LDDTree &Dependencies,
             const std::string &Path,
             unsigned DepthLevel) {
  if (DepthLevel > 0)
    lddtreeHelper(Dependencies, Path, 1, DepthLevel);
}
