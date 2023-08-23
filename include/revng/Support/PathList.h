#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <map>
#include <optional>
#include <string>
#include <vector>

#include "llvm/ADT/SmallString.h"
#include "llvm/Support/Path.h"

llvm::StringRef getCurrentRoot();
const std::map<std::string, std::string> &getLibrariesFullPath();

template<typename... T>
  requires(std::is_convertible_v<T, llvm::StringRef> && ...)
std::string joinPath(const llvm::StringRef First, const T... Parts) {
  llvm::SmallString<128> ResultPath(First);
  (llvm::sys::path::append(ResultPath, Parts), ...);
  return ResultPath.str().str();
}

class PathList {
public:
  PathList(const std::vector<std::string> &Paths) : SearchPaths(Paths) {}

  std::optional<std::string> findFile(llvm::StringRef FileName) const;
  std::vector<std::string> list(llvm::StringRef Path,
                                llvm::StringRef Suffix) const;

private:
  std::vector<std::string> SearchPaths;
};
