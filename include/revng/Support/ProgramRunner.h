#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <string>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

class ProgramRunner {
private:
  llvm::SmallVector<std::string, 64> Paths;
  llvm::SmallVector<llvm::StringRef, 64> PathsRef;

public:
  ProgramRunner();

  /// Returns true if the program could be found.
  bool isProgramAvailable(llvm::StringRef ProgramName);

  /// returns the exit code of the program.
  [[nodiscard]] int run(llvm::StringRef ProgramName,
                        llvm::ArrayRef<std::string> Args);
};

extern ProgramRunner Runner;
