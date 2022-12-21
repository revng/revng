#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <optional>
#include <string>
#include <vector>

std::string getCurrentExecutableFullPath();
std::string getCurrentRoot();

class PathList {
public:
  PathList(const std::vector<std::string> &Paths) : SearchPaths(Paths) {}

  std::optional<std::string> findFile(llvm::StringRef FileName) const;

private:
  std::vector<std::string> SearchPaths;
};
