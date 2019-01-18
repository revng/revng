/// \file debug.cpp
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

cl::opt<bool> UseDebugSymbols("use-debug-symbols",
                              cl::desc("use section and symbol function "
                                       "informations, if available"),
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
