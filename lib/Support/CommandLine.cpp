/// \file CommandLine.cpp

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <fstream>
#include <iostream>

#include "revng/Support/CommandLine.h"

namespace cl = llvm::cl;

cl::OptionCategory MainCategory("Options", "");

std::ostream &pathToStream(const std::string &Path, std::ofstream &File) {
  if (Path[0] == '-' && Path[1] == '\0') {
    return std::cout;
  } else {
    if (File.is_open())
      File.close();
    File.open(Path);
    return File;
  }
}
