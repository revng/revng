//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

extern "C" {
#include "dlfcn.h"
#include "link.h"
}

#include <map>
#include <set>

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/GlobPattern.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"

#include "revng/Support/Assert.h"
#include "revng/Support/Debug.h"
#include "revng/Support/PathList.h"

static Logger<> Log("find-resources");

struct Resources {
  bool FirstCall = true;
  std::string CurrentRoot;
  std::map<std::string, std::string> LibrariesFullPath;
};

llvm::ManagedStatic<Resources> Resources;

extern "C" {
static int
dlIteratePhdrCallback(struct dl_phdr_info *Info, size_t Size, void *Data) {
  using llvm::StringRef;

  if (Info->dlpi_name == nullptr)
    return 0;

  StringRef FullPath(Info->dlpi_name);
  if (FullPath.size() == 0)
    return 0;

  StringRef Name = llvm::sys::path::filename(FullPath);
  Resources->LibrariesFullPath[Name.rsplit(".so").first.str()] = FullPath.str();

  return 0;
}
}

static void initialize() {
  if (not Resources->FirstCall)
    return;

  Resources->FirstCall = false;

  dl_iterate_phdr(dlIteratePhdrCallback, nullptr);

  using namespace llvm::sys::path;
  constexpr const char *MainLibrary = "librevngSupport";
  using llvm::StringRef;
  StringRef MainLibraryFullPath = Resources->LibrariesFullPath.at(MainLibrary);
  Resources->CurrentRoot = parent_path(parent_path(MainLibraryFullPath));
}

llvm::StringRef getCurrentRoot() {
  initialize();
  return Resources->CurrentRoot;
}

const std::map<std::string, std::string> &getLibrariesFullPath() {
  initialize();
  return Resources->LibrariesFullPath;
}

static std::optional<std::string>
findFileInPaths(llvm::StringRef FileName,
                const std::vector<std::string> &SearchPaths) {

  std::optional<std::string> FoundFileName;

  for (const auto &Path : SearchPaths) {
    revng_log(Log, "Looking in path: " << Path);
    llvm::SmallString<64> FullFileName;
    llvm::sys::path::append(FullFileName, Path, FileName);

    LoggerIndent<> Indent(Log);

    if (not llvm::sys::fs::exists(FullFileName)) {
      revng_log(Log, "File not found: " << FullFileName.str().str());
      continue;
    }

    std::error_code
      Err = llvm::sys::fs::access(FullFileName,
                                  llvm::sys::fs::AccessMode::Exist);
    if (Err) {
      revng_log(Log, "Cannot access file: " << FullFileName.str().str());
    } else {
      revng_log(Log, "Found file: " << FullFileName.str().str());
      FoundFileName = FullFileName.str().str();
      break;
    }
  }

  revng_assert(FoundFileName,
               ("failed to find '" + FileName + "'.").str().c_str());

  return FoundFileName;
}

std::optional<std::string> PathList::findFile(llvm::StringRef FileName) const {
  return findFileInPaths(FileName, SearchPaths);
}

std::vector<std::string> PathList::list(llvm::StringRef Path,
                                        llvm::StringRef Suffix) const {
  using namespace llvm;
  using namespace sys;
  using namespace fs;
  using std::string;

  std::set<string> Visited;
  std::vector<string> Result;
  for (const string &SearchPath : SearchPaths) {
    SmallString<16> FullPath(SearchPath);
    path::append(FullPath, Path);

    if (is_directory(FullPath)) {
      std::error_code EC;
      for (directory_iterator File(FullPath, EC), FileEnd;
           File != FileEnd && !EC;
           File.increment(EC)) {
        if (llvm::StringRef(File->path()).endswith(Suffix)) {
          bool New = Visited.insert(path::filename(File->path()).str()).second;
          if (New)
            Result.push_back(File->path());
        }
      }

      revng_assert(!EC);
    }
  }

  return Result;
}
