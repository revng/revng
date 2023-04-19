/// \file LDDTree.cpp
/// \brief Implementation of lddtree like API

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

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
#include "revng/Support/Debug.h"
#include "revng/Support/LDDTree.h"

Logger<> Log("lddtree");

using namespace llvm;

static llvm::Expected<std::string>
findInDir(llvm::StringRef Lib, llvm::StringRef WhereToLook) {
  llvm::SmallString<128> ResultPath;
  llvm::sys::path::append(ResultPath, WhereToLook, Lib);

  if (llvm::sys::fs::exists(ResultPath.str()))
    return std::string(ResultPath.str());

  std::error_code EC;
  llvm::sys::fs::directory_iterator DirIt(WhereToLook, EC);
  llvm::sys::fs::directory_iterator DirEnd;
  if (EC) {
    revng_log(Log,
              "Unable to find library: "
                << toString(llvm::errorCodeToError(EC)));
    return llvm::errorCodeToError(EC);
  }
  while (DirIt != DirEnd) {
    ResultPath.clear();
    llvm::sys::path::append(ResultPath, DirIt->path(), Lib);

    DirIt.increment(EC);
    if (EC) {
      revng_log(Log,
                "Unable to find library: "
                  << toString(llvm::errorCodeToError(EC)));
      return llvm::errorCodeToError(EC);
    }

    if (llvm::sys::fs::exists(ResultPath.str()))
      return std::string(ResultPath.str());
  }

  revng_log(Log, "Unable to find " << Lib);
  return std::string();
}

static std::optional<std::string>
findLibrary(llvm::StringRef Lib, std::optional<llvm::StringRef> RunPath) {
  llvm::SmallString<128> LibraryPath;

  // First process the `LD_LIBRARY_PATH`.
  auto LDLibraryPath = llvm::sys::Process::GetEnv("LD_LIBRARY_PATH");
  if (LDLibraryPath) {
    StringRef TheLDLibraryPath(*LDLibraryPath);
    SmallVector<StringRef, 16> LibraryPaths;
    TheLDLibraryPath.split(LibraryPaths, ":");
    for (auto Path : LibraryPaths) {
      llvm::sys::path::append(LibraryPath, Path, Lib);
      if (llvm::sys::fs::exists(LibraryPath))
        return std::string(LibraryPath.str());
      LibraryPath.clear();
    }
  }

  // Process -rpath.
  if (RunPath and (*RunPath).size()) {
    SmallVector<StringRef, 16> LibraryRunPaths;
    (*RunPath).split(LibraryRunPaths, ":");
    for (auto Path : LibraryRunPaths) {
      llvm::sys::path::append(LibraryPath, Path, Lib);

      if (llvm::sys::fs::exists(LibraryPath))
        return std::string(LibraryPath.str());
      LibraryPath.clear();
    }
  }

  // Try in /lib/.
  auto LibraryAsCanonical = findInDir(Lib, "/lib/");
  if (!LibraryAsCanonical) {
    revng_log(Log,
              "Unable to find library: "
                << toString(LibraryAsCanonical.takeError()));
    llvm::consumeError(LibraryAsCanonical.takeError());
    return std::nullopt;
  }

  if ((*LibraryAsCanonical).size())
    return *LibraryAsCanonical;

  LibraryAsCanonical = findInDir(Lib, "/usr/lib/");
  if (!LibraryAsCanonical) {
    revng_log(Log,
              "Unable to find library: "
                << toString(LibraryAsCanonical.takeError()));
    llvm::consumeError(LibraryAsCanonical.takeError());
    return std::nullopt;
  }

  if ((*LibraryAsCanonical).size())
    return *LibraryAsCanonical;

  LibraryAsCanonical = findInDir(Lib, "/usr/local/lib/");
  if (!LibraryAsCanonical) {
    revng_log(Log,
              "Unable to find library: "
                << toString(LibraryAsCanonical.takeError()));
    llvm::consumeError(LibraryAsCanonical.takeError());
    return std::nullopt;
  }

  if ((*LibraryAsCanonical).size())
    return *LibraryAsCanonical;

  return std::nullopt;
}

template<class ELFT>
std::optional<llvm::StringRef>
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
void lddtreeResolve(LDDTree &Dependencies,
                    StringRef FileName,
                    const llvm::object::ELFFile<ELFT> &TheELF) {
  llvm::StringRef DynamicStringTable;

  using Elf_Shdr = const typename llvm::object::ELFFile<ELFT>::Elf_Shdr;
  typename ELFT::ShdrRange Sections = cantFail(TheELF.sections());
  for (const Elf_Shdr &Sec : Sections) {
    if (Sec.sh_type == llvm::ELF::SHT_DYNSYM) {
      if (auto E = TheELF.getStringTableForSymtab(Sec)) {
        DynamicStringTable = *E;
        break;
      }
    }
  }

  auto DynamicEntries = TheELF.dynamicEntries();
  if (!DynamicEntries)
    return;

  std::optional<StringRef> RunPath;

  using Elf_Dyn = const typename llvm::object::ELFFile<ELFT>::Elf_Dyn;

  // Find RUNPATH generated with -rpath.
  for (Elf_Dyn &DynamicTag : *DynamicEntries) {
    auto TheTag = DynamicTag.getTag();
    auto TheVal = DynamicTag.getVal();
    if (TheTag == llvm::ELF::DT_RUNPATH) {
      RunPath = getDynamicString<ELFT>(TheELF, DynamicStringTable, TheVal);
      if (RunPath)
        revng_log(Log, "  RUNPATH: " << *RunPath);
      else
        revng_log(Log, "  Unable to parse RUNPATH");
      break;
    }
  }

  for (Elf_Dyn &DynamicTag : *DynamicEntries) {
    auto TheTag = DynamicTag.getTag();
    auto TheVal = DynamicTag.getVal();
    if (TheTag == llvm::ELF::DT_NEEDED) {
      auto LibName = getDynamicString<ELFT>(TheELF, DynamicStringTable, TheVal);
      if (!LibName) {
        revng_log(Log, "  Unable to parse needed library name");
        continue;
      }
      revng_log(Log, "  - needeed library: " << *LibName << "\n");
      if (auto LocOfLib = findLibrary(*LibName, RunPath))
        Dependencies[FileName.str()].push_back(*LocOfLib);
    }
  }
}

static RecursiveCoroutine<void> lddtreeHelper(LDDTree &Dependencies,
                                              const std::string &Path,
                                              unsigned CurrentLevel,
                                              unsigned DepthLevel) {
  revng_log(Log, "lddtree for " << Path << "\n");
  auto BinaryOrErr = object::createBinary(Path);
  if (not BinaryOrErr) {
    revng_log(Log,
              "Can't create binary: " << toString(BinaryOrErr.takeError()));
    llvm::consumeError(BinaryOrErr.takeError());
    rc_return;
  }
  auto &Obj = *cast<object::ObjectFile>(BinaryOrErr->getBinary());
  if (auto *ELFObj = dyn_cast<object::ELF32LEObjectFile>(&Obj))
    lddtreeResolve(Dependencies, Path, ELFObj->getELFFile());
  else if (auto *ELFObj = dyn_cast<object::ELF32BEObjectFile>(&Obj))
    lddtreeResolve(Dependencies, Path, ELFObj->getELFFile());
  else if (auto *ELFObj = dyn_cast<object::ELF64LEObjectFile>(&Obj))
    lddtreeResolve(Dependencies, Path, ELFObj->getELFFile());
  else if (auto *ELFObj = dyn_cast<object::ELF64BEObjectFile>(&Obj))
    lddtreeResolve(Dependencies, Path, ELFObj->getELFFile());
  else
    revng_log(Log, "Not an ELF.");

  if (CurrentLevel == DepthLevel)
    rc_return;

  ++CurrentLevel;
  for (auto &I : Dependencies) {
    revng_log(Log, "Dependencies for " << I.first << ":\n");
    for (auto &J : I.second)
      if (!Dependencies.count(J))
        rc_recur lddtreeHelper(Dependencies, J, CurrentLevel, DepthLevel);
  }
}

void lddtree(LDDTree &Dependencies,
             const std::string &Path,
             unsigned DepthLevel) {
  if (DepthLevel > 0)
    lddtreeHelper(Dependencies, Path, 1, DepthLevel);
}
