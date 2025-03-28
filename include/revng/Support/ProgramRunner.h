#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <string>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

class ProgramRunner {
public:
  struct Result {
    /// The exit code of the program, always present
    int ExitCode = -1;
    /// Captured stdout, if RunOptions::Capture was specified to do it,
    /// otherwise an empty string
    std::string Stdout;
    /// Captured stderr, if RunOptions::Capture was specified to do it,
    /// otherwise an empty string
    std::string Stderr;
  };

  enum class CaptureOption {
    /// No capure
    None,
    /// Capture only stdout
    StdoutOnly,
    /// Capture only stderr
    StderrOnly,
    /// Capture both stdout and stderr separately
    StdoutAndStderrSeparately,
    /// Capture stdout, redirect stderr to stdout
    StdoutAndStderrJoined,
  };

  struct RunOptions {
    /// If provided, will pipe the provided string to stdin
    std::optional<std::string> Stdin;
    /// Capture output options
    CaptureOption Capture = CaptureOption::None;
  };

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

  /// returns a Result object after running the program.
  [[nodiscard]] Result run(llvm::StringRef ProgramName,
                           llvm::ArrayRef<std::string> Args,
                           RunOptions Options);
};

extern ProgramRunner Runner;
