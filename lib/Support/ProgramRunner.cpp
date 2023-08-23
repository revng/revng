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
#include "llvm/Support/Program.h"

#include "revng/Support/Assert.h"
#include "revng/Support/PathList.h"
#include "revng/Support/ProgramRunner.h"

using namespace llvm;

ProgramRunner Runner;

ProgramRunner::ProgramRunner() {
  using namespace llvm::sys::path;
  llvm::SmallString<128> BinPath;
  append(BinPath, getCurrentRoot(), "bin");
  Paths = { BinPath.str().str() };

  // Append PATH
  char *Path = getenv("PATH");
  if (Path == nullptr)
    return;
  llvm::SmallVector<llvm::StringRef, 64> BasePaths;
  StringRef(Path).split(BasePaths, ":");
  for (llvm::StringRef BasePath : BasePaths)
    Paths.push_back(BasePath.str());
}

bool ProgramRunner::isProgramAvailable(llvm::StringRef ProgramName) {
  llvm::SmallVector<llvm::StringRef, 64> PathsRef;
  for (const std::string &Path : Paths)
    PathsRef.push_back(llvm::StringRef(Path));

  auto MaybeProgramPath = llvm::sys::findProgramByName(ProgramName, PathsRef);
  if (!MaybeProgramPath)
    return false;
  return true;
}

int ProgramRunner::run(llvm::StringRef ProgramName,
                       ArrayRef<std::string> Args) {
  llvm::SmallVector<llvm::StringRef, 64> PathsRef;
  for (const std::string &Path : Paths)
    PathsRef.push_back(llvm::StringRef(Path));

  auto MaybeProgramPath = llvm::sys::findProgramByName(ProgramName, PathsRef);
  revng_assert(not Paths.empty());
  revng_assert(MaybeProgramPath,
               (ProgramName + " was not found in " + getenv("PATH"))
                 .str()
                 .c_str());

  // Prepare actual arguments
  std::vector<StringRef> StringRefs{ *MaybeProgramPath };
  for (const std::string &Arg : Args)
    StringRefs.push_back(Arg);

  int ExitCode = llvm::sys::ExecuteAndWait(StringRefs[0], StringRefs);

  return ExitCode;
}
