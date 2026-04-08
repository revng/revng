//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/ADT/SmallString.h"
#include "llvm/BinaryFormat/ELF.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Process.h"

#include "revng/Support/ObjectFile.h"
#include "revng/Support/OverflowSafeInt.h"

#include "LDDTreeObjectFileTraits.h"

using namespace llvm;
using namespace llvm::object;

template<class ELFT>
std::optional<std::string> getDynamicString(const ELFFile<ELFT> &TheELF,
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
  return StringRef(DynamicStringTable.data() + Value).str();
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
  auto MaybeProgramHeaders = ELFObjectFile.getELFFile().program_headers();
  if (auto Error = MaybeProgramHeaders.takeError()) {
    revng_log(LDDTreeLog, "No valid program headers available: " << Error);
    llvm::consumeError(std::move(Error));
    return {};
  }

  // Find the correct program header
  using Elf_Phdr = ELFT::Elf_Phdr;
  for (const Elf_Phdr &Phdr : *MaybeProgramHeaders) {
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

template<IsELFObjectFile T>
class ELFObjectFileTraits {
public:
  using ObjectFile = T;

  struct ResolutionInfo {
    bool Is64 = false;
    bool NoDefault = false;
    uint16_t EMachine = 0;
    std::optional<std::string> RPath;
    std::optional<std::string> RunPath;
  };

  using QueueEntry = typename Queue<ResolutionInfo>::Entry;

public:
  static std::vector<std::pair<llvm::StringRef, uint64_t>>
  getExportedSymbols(const T &ELFObjectFile) {
    std::vector<std::pair<llvm::StringRef, uint64_t>> Result;

    for (object::ELFSymbolRef Symbol :
         llvm::make_range(ELFObjectFile.dynamic_symbol_begin(),
                          ELFObjectFile.dynamic_symbol_end())) {

      // TODO: consider global objects as well
      if (Symbol.getELFType() != ELF::STT_FUNC
          and Symbol.getELFType() != ELF::STT_GNU_IFUNC)
        continue;

      auto MaybeName = Symbol.getName();
      if (auto Error = MaybeName.takeError()) {
        revng_log(LDDTreeLog, "Couldn't get the name of a dynamic symbol");
        continue;
      }

      auto MaybeSection = Symbol.getSection();
      if (auto Error = MaybeSection.takeError()) {
        revng_log(LDDTreeLog,
                  "Couldn't get the section of dynamic symbol " << *MaybeName);
        continue;
      }

      // Skip undefined symbols
      if (*MaybeSection == ELFObjectFile.section_end())
        continue;

      auto MaybeAddress = Symbol.getAddress();
      if (auto Error = MaybeAddress.takeError()) {
        revng_log(LDDTreeLog,
                  "Couldn't get the address of dynamic symbol " << *MaybeName);
        continue;
      }

      Result.push_back({ *MaybeName, *MaybeAddress });
    }

    return Result;
  }

  static std::vector<QueueEntry>
  getDependencies(const revng::RootEntry &Root,
                  StringRef ImporterIdentifier,
                  StringRef ImporterCanonicalPath,
                  const ObjectFile &ELFObjectFile) {
    std::vector<QueueEntry> Results;
    const auto &TheELF = ELFObjectFile.getELFFile();

    auto MaybeDynamicEntries = TheELF.dynamicEntries();

    if (auto Error = MaybeDynamicEntries.takeError()) {
      revng_log(LDDTreeLog, "Cannot access dynamic entries: " << Error);
      consumeError(std::move(Error));
      return {};
    }

    using Elf_Dyn_Range = ObjectFile::Elf_Dyn_Range;
    Elf_Dyn_Range DynamicEntries = *MaybeDynamicEntries;

    std::optional<std::string> RunPath;
    std::optional<std::string> RPath;
    bool NoDefault = false;

    // Look for .dynstr
    StringRef DynamicStringTable;
    if (auto MaybeDynamicStringTable = getDynamicStringTable(ELFObjectFile,
                                                             DynamicEntries))
      DynamicStringTable = *MaybeDynamicStringTable;

    if (DynamicStringTable.empty()) {
      revng_log(LDDTreeLog, "Cannot find .dynstr");
      return {};
    }

    // Look for DT_RPATH and DT_RUNPATH
    using Elf_Dyn = ObjectFile::Elf_Dyn;
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

    bool Is64 = (std::is_same_v<ObjectFile, object::ELF64LEObjectFile>
                 or std::is_same_v<ObjectFile, object::ELF64BEObjectFile>);

    for (const Elf_Dyn &DynamicTag : DynamicEntries) {
      auto TheTag = DynamicTag.getTag();
      auto TheVal = DynamicTag.getVal();
      if (TheTag == llvm::ELF::DT_NEEDED) {
        auto LibName = getDynamicString(TheELF, DynamicStringTable, TheVal);
        if (!LibName) {
          revng_log(LDDTreeLog, "Unable to parse needed library name");
          continue;
        }

        Results.push_back({ *LibName,
                            ImporterIdentifier.str(),
                            ImporterCanonicalPath.str(),
                            { Is64,
                              NoDefault,
                              TheELF.getHeader().e_machine,
                              RPath,
                              RunPath } });
      }
    }

    return Results;
  }

  /// \see man ld.so
  static SmallVector<std::string, 16>
  computeSearchPaths(const revng::RootEntry &Root, QueueEntry &Entry) {
    // Note: despite taking Root, the results should not be prefixed with root!

    auto &Dependency = Entry.Dependency;

    SmallVector<std::string, 16> SearchPaths;

    // Process DT_RPATH
    if (Dependency.RPath and Dependency.RPath->size())
      for (StringRef Path : split(*Dependency.RPath, ":"))
        SearchPaths.push_back(Path.str());

    using namespace llvm::sys;
#ifdef __linux__
    // Process the `LD_LIBRARY_PATH`
    if (Root.isHost())
      if (auto MaybeLibraryPath = Process::GetEnv("LD_LIBRARY_PATH"))
        for (StringRef Path : split(*MaybeLibraryPath, ":"))
          SearchPaths.push_back(Path.str());
#endif

    // Process DT_RUNPATH
    std::string Origin;
    {
      SmallString<32> OriginSmallString;
      OriginSmallString += Entry.ImporterCanonicalPath;
      path::remove_filename(OriginSmallString);
      Origin = OriginSmallString.str();
    }

    if (Origin.size() == 0)
      Origin = path::get_separator();

    std::string LibName = Dependency.Is64 ? "lib64" : "lib";
    if (Dependency.RunPath and Dependency.RunPath->size()) {
      for (StringRef Path : split(*Dependency.RunPath, ":")) {
        std::string PathString = Path.str();
        replaceAll(PathString, "$ORIGIN", Origin);
        replaceAll(PathString, "${ORIGIN}", Origin);
        replaceAll(PathString, "$LIB", LibName);
        replaceAll(PathString, "${LIB}", LibName);
        // TODO: handle $PLATFORM

        SearchPaths.push_back(PathString);
      }
    }

    if (not Dependency.NoDefault) {
      auto &LinuxRoot = Root.osSpecific<revng::LinuxRootFS>();
      LinuxRoot.addLdSoConfPaths(SearchPaths);
      SearchPaths.push_back("/" + LibName);
      SearchPaths.push_back("/usr/" + LibName);
    }

    return SearchPaths;
  }

  static bool isValidObjectFor(QueueEntry &Entry, const T &Elf) {
    auto &Dependency = Entry.Dependency;
    auto Machine = static_cast<const ELFObjectFileBase &>(Elf).getEMachine();

    if (Machine != Dependency.EMachine) {
      revng_log(LDDTreeLog,
                "Found candidate but it has the wrong e_machine: "
                  << Machine << " (expected " << Dependency.EMachine << ").");
      return false;
    }

    return true;
  }
};

template<>
class LDDTreeObjectFileTraits<ELF32LEObjectFile>
  : public ELFObjectFileTraits<ELF32LEObjectFile> {};

template<>
class LDDTreeObjectFileTraits<ELF32BEObjectFile>
  : public ELFObjectFileTraits<ELF32BEObjectFile> {};

template<>
class LDDTreeObjectFileTraits<ELF64LEObjectFile>
  : public ELFObjectFileTraits<ELF64LEObjectFile> {};

template<>
class LDDTreeObjectFileTraits<ELF64BEObjectFile>
  : public ELFObjectFileTraits<ELF64BEObjectFile> {};

template<typename T>
concept Check = LDDTreeObjectFileTraitsConcept<LDDTreeObjectFileTraits<T>>;

static_assert(Check<ELF32LEObjectFile>);
static_assert(Check<ELF64LEObjectFile>);
static_assert(Check<ELF32BEObjectFile>);
static_assert(Check<ELF64BEObjectFile>);

template const LDDTree
LDDTree::fromPath<ELF32LEObjectFile>(const revng::RootEntry &Root,
                                     StringRef Path,
                                     const ELF32LEObjectFile &Binary,
                                     std::optional<LDDTree::SymbolSet> Symbols);

template const LDDTree
LDDTree::fromPath<ELF64LEObjectFile>(const revng::RootEntry &Root,
                                     StringRef Path,
                                     const ELF64LEObjectFile &Binary,
                                     std::optional<LDDTree::SymbolSet> Symbols);

template const LDDTree
LDDTree::fromPath<ELF32BEObjectFile>(const revng::RootEntry &Root,
                                     StringRef Path,
                                     const ELF32BEObjectFile &Binary,
                                     std::optional<LDDTree::SymbolSet> Symbols);

template const LDDTree
LDDTree::fromPath<ELF64BEObjectFile>(const revng::RootEntry &Root,
                                     StringRef Path,
                                     const ELF64BEObjectFile &Binary,
                                     std::optional<LDDTree::SymbolSet> Symbols);
