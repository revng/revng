/// \file CommandLine.cpp
/// \brief Implementation of the debug framework

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// Standard includes
#include <fstream>
#include <iostream>

// Local libraries includes
#include "revng/Support/CommandLine.h"

namespace cl = llvm::cl;

cl::OptionCategory MainCategory("Options", "");

cl::opt<bool> IgnoreDebugSymbols("ignore-debug-symbols",
                                 cl::desc("ignore section and symbol function "
                                          "informations"),
                                 cl::cat(MainCategory));

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
