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
inline std::string joinPath(llvm::sys::path::Style Style,
                            const llvm::StringRef First,
                            const T... Parts) {
  llvm::SmallString<128> ResultPath(First);
  (llvm::sys::path::append(ResultPath, Style, Parts), ...);
  return ResultPath.str().str();
}

template<typename... T>
  requires(std::is_convertible_v<T, llvm::StringRef> && ...)
inline std::string joinPath(const llvm::StringRef First, const T... Parts) {
  return joinPath(llvm::sys::path::Style::native, First, Parts...);
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
