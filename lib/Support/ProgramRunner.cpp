/// \file ProgramRunner.cpp
/// A program runner is used to invoke external programs a bit more safelly than
/// to compose a string and invoke system.

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <cstdlib>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Process.h"
#include "llvm/Support/Program.h"

#include "revng/Support/Assert.h"
#include "revng/Support/Debug.h"
#include "revng/Support/PathList.h"
#include "revng/Support/ProgramRunner.h"
#include "revng/Support/TemporaryFile.h"

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
  return run(ProgramName, Args, {}).ExitCode;
}

ProgramRunner::Result ProgramRunner::run(llvm::StringRef ProgramName,
                                         llvm::ArrayRef<std::string> Args,
                                         RunOptions Options) {
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

  // Prepare redirects
  std::optional<TemporaryFile> Stdin;
  std::optional<TemporaryFile> Stdout;
  std::optional<TemporaryFile> Stderr;
  std::vector<std::optional<llvm::StringRef>> Redirects = { std::nullopt,
                                                            std::nullopt,
                                                            std::nullopt };
  if (Options.Stdin.has_value()) {
    Stdin.emplace("ProgramRunner-stdin");
    Redirects[0] = Stdin->path();

    std::error_code EC;
    llvm::raw_fd_stream Stream(Stdin->path(), EC);
    revng_assert(not EC);
    Stream << *(Options.Stdin);
    Stream.flush();
  }

  if (Options.Capture == CaptureOption::StdoutOnly
      || Options.Capture == CaptureOption::StdoutAndStderrSeparately
      || Options.Capture == CaptureOption::StdoutAndStderrJoined) {
    Stdout.emplace("ProgramRunner-stdout");
    Redirects[1] = Stdout->path();
    if (Options.Capture == CaptureOption::StdoutAndStderrJoined)
      Redirects[2] = Stdout->path();
  }

  if (Options.Capture == CaptureOption::StderrOnly
      || Options.Capture == CaptureOption::StdoutAndStderrSeparately) {
    Stderr.emplace("ProgramRunner-stderr");
    Redirects[2] = Stderr->path();
  }

  int ExitCode = llvm::sys::ExecuteAndWait(StringRefs[0],
                                           StringRefs,
                                           /* Env */ std::nullopt,
                                           Redirects);
  revng_log(Log, "Program exited with code " << ExitCode);

  std::string StdoutString;
  std::string StderrString;
  if (Stdout.has_value()) {
    auto Buffer = revng::cantFail(llvm::MemoryBuffer::getFile(Stdout->path()));
    StdoutString = Buffer->getBuffer().str();
  }

  if (Stderr.has_value()) {
    auto Buffer = revng::cantFail(llvm::MemoryBuffer::getFile(Stderr->path()));
    StderrString = Buffer->getBuffer().str();
  }

  return { ExitCode, StdoutString, StderrString };
}
