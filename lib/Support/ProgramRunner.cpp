#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Program.h"

#include "revng/Support/Assert.h"
#include "revng/Support/ProgramRunner.h"

using namespace llvm;
using namespace sys;

ProgramRunner Runner;

ProgramRunner::ProgramRunner() :
  MainExecutable(fs::getMainExecutable("", reinterpret_cast<void *>(main))) {
  Paths = { { path::parent_path(MainExecutable) } };

  // Append PATH
  char *Path = getenv("PATH");
  if (Path == nullptr)
    return;
  StringRef(Path).split(Paths, ":");
}

void ProgramRunner::run(llvm::StringRef ProgramName,
                        std::vector<std::string> &Args) {
  auto MaybeProgramPath = findProgramByName(ProgramName, Paths);
  revng_assert(MaybeProgramPath);

  // Prepare actual arguments
  std::vector<StringRef> StringRefs{ *MaybeProgramPath };
  for (std::string &Arg : Args)
    StringRefs.push_back(Arg);

  // Invoke linker
  int ExitCode = ExecuteAndWait(StringRefs[0], StringRefs);
  revng_assert(ExitCode == 0);
}
