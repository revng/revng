#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <string>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

extern int main(int argc, char *argv[]);

class ProgramRunner {
private:
  std::string MainExecutable;
  llvm::SmallVector<llvm::StringRef, 64> Paths;

public:
  ProgramRunner();

  void run(llvm::StringRef ProgramName, std::vector<std::string> &Args);
};

extern ProgramRunner Runner;
