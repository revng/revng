//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

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

std::string getCurrentExecutableFullPath() {
  // TODO: make this optional
  llvm::SmallString<128> SelfPath("/proc/self/exe");
  llvm::SmallString<128> FullPath;
  std::error_code Err = llvm::sys::fs::real_path(SelfPath, FullPath);
  if (Err)
    revng_abort("Could not determine executable path");

  revng_assert(not FullPath.empty());

  return FullPath.str().str();
}

std::string getCurrentRoot() {
  using llvm::sys::path::parent_path;

  auto MaybeBuffer = llvm::MemoryBuffer::getFileAsStream("/proc/self/maps");
  if (!MaybeBuffer)
    revng_abort("Unable to open /proc/self/maps");

  auto Maps = (*MaybeBuffer)->getBuffer();
  llvm::SmallVector<llvm::StringRef, 128> MapsLines;
  Maps.split(MapsLines, "\n");

  for (const auto &Line : MapsLines) {
    llvm::SmallVector<llvm::StringRef> Parts;
    Line.split(Parts, ' ', 5);
    revng_assert(Parts.size() == 6, "Malformed /proc/self/maps");

    llvm::StringRef File = Parts[5].ltrim();

    if (File.endswith("librevngSupport.so")) {
      llvm::SmallString<128> FullPath;

      std::error_code Err = llvm::sys::fs::real_path(File, FullPath);
      if (Err)
        revng_abort("Could not find real path of librevngSupport.so");

      revng_assert(!FullPath.empty());

      return parent_path(parent_path(FullPath)).str();
    }
  }
  revng_abort("Could not determine root folder");
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

std::vector<std::string>
PathList::list(llvm::StringRef Path, llvm::StringRef Suffix) const {
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
