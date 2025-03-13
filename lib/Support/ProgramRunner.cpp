/// \file ProgramRunner.cpp
/// A program runner is used to invoke external programs a bit more safelly than
/// to compose a string and invoke system.

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <cstdlib>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Process.h"
#include "llvm/Support/Program.h"

#include "revng/Support/Assert.h"
#include "revng/Support/Debug.h"
#include "revng/Support/PathList.h"
#include "revng/Support/ProgramRunner.h"

static Logger<> Log("program-runner");

ProgramRunner Runner;

ProgramRunner::ProgramRunner() {
  llvm::SmallString<128> BinPath;
  llvm::sys::path::append(BinPath, getCurrentRoot(), "bin");
  Paths = { BinPath.str().str() };

  // Append PATH
  std::optional<std::string> Path = llvm::sys::Process::GetEnv("PATH");
  if (not Path.has_value())
    return;

  llvm::SmallVector<llvm::StringRef, 64> BasePaths;
  llvm::StringRef(*Path).split(BasePaths, llvm::sys::EnvPathSeparator);
  for (const llvm::StringRef &BasePath : BasePaths)
    Paths.push_back(BasePath.str());

  PathsRef = { Paths.begin(), Paths.end() };
}

bool ProgramRunner::isProgramAvailable(llvm::StringRef ProgramName) {
  return static_cast<bool>(llvm::sys::findProgramByName(ProgramName, PathsRef));
}

int ProgramRunner::run(llvm::StringRef ProgramName,
                       llvm::ArrayRef<std::string> Args) {
  auto MaybeProgramPath = llvm::sys::findProgramByName(ProgramName, PathsRef);
  revng_assert(MaybeProgramPath,
               (ProgramName.str() + " was not found in "
                + llvm::join(Paths,
                             llvm::StringRef(&llvm::sys::EnvPathSeparator, 1)))
                 .c_str());

  // Prepare actual arguments
  std::vector<llvm::StringRef> StringRefs{ *MaybeProgramPath };
  for (const std::string &Arg : Args)
    StringRefs.push_back(Arg);

  if (Log.isEnabled()) {
    Log << "Running " << StringRefs[0] << " with the following arguments:\n";
    for (const std::string &Arg : Args)
      Log << "  " << Arg << "\n";
    Log << DoLog;
  }

  int ExitCode = llvm::sys::ExecuteAndWait(StringRefs[0], StringRefs);
  revng_log(Log, "Program exited with code " << ExitCode);

  return ExitCode;
}
