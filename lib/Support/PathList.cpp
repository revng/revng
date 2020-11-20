//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/Support/Assert.h"
#include "revng/Support/Debug.h"
#include "revng/Support/PathList.h"

static Logger<> Log("find-resources");

std::string getCurrentExecutableFullPath() {

  std::string Result;

  // TODO: make this optional
  llvm::SmallString<128> SelfPath("/proc/self/exe");
  llvm::SmallString<128> FullPath;
  std::error_code Err = llvm::sys::fs::real_path(SelfPath, FullPath);
  if (Err)
    return Result;

  revng_assert(not FullPath.empty());

  llvm::sys::path::remove_filename(FullPath);
  Result = FullPath.str();
  return Result;
}

static std::optional<std::string>
findFileInPaths(const std::string &FileName,
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

  return FoundFileName;
}

std::optional<std::string>
PathList::findFile(const std::string &FileName) const {
  return findFileInPaths(FileName, SearchPaths);
}
