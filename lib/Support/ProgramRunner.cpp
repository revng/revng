/// \file ProgramRunner.cpp
/// \brief a program runner is used to invoke external programs a bit more
/// safelly than to compose a string and invoke system

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Program.h"

#include "revng/Support/Assert.h"
#include "revng/Support/PathList.h"
#include "revng/Support/ProgramRunner.h"

using namespace llvm;
using namespace sys;

ProgramRunner Runner;

ProgramRunner::ProgramRunner() {
  Paths = { { path::parent_path(getCurrentExecutableFullPath()) } };

  // Append PATH
  char *Path = getenv("PATH");
  if (Path == nullptr)
    return;
  StringRef(Path).split(Paths, ":");
}

int ProgramRunner::run(llvm::StringRef ProgramName,
                       ArrayRef<std::string> Args) {
  auto MaybeProgramPath = findProgramByName(ProgramName, Paths);
  revng_assert(MaybeProgramPath);

  // Prepare actual arguments
  std::vector<StringRef> StringRefs{ *MaybeProgramPath };
  for (const std::string &Arg : Args)
    StringRefs.push_back(Arg);

  int ExitCode = ExecuteAndWait(StringRefs[0], StringRefs);

  return ExitCode;
}
