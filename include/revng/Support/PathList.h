#ifndef REVNG_PATH_LIST
#define REVNG_PATH_LIST

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <optional>
#include <string>
#include <vector>

std::string getCurrentExecutableFullPath();

class PathList {
public:
  PathList(const std::vector<std::string> &Paths) : SearchPaths(Paths) {}

  std::optional<std::string> findFile(const std::string &FileName) const;

private:
  std::vector<std::string> SearchPaths;
};

#endif // REVNG_PATH_LIST
