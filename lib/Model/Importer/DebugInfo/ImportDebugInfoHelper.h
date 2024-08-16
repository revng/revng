#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/Support/Path.h"
#include "llvm/Support/Process.h"

#include "revng/Support/Assert.h"
#include "revng/Support/ProgramRunner.h"

namespace {

inline int runFetchDebugInfo(llvm::StringRef InputFileName) {
  revng_assert(::Runner.isProgramAvailable("revng"));
  return ::Runner.run("revng",
                      { "model", "fetch-debuginfo", InputFileName.str() });
}

inline void setXDG(llvm::SmallVectorImpl<char> &Destination,
                   const llvm::Twine &XDGVariable,
                   const llvm::Twine &Default) {
  using namespace llvm;
  revng_assert(Destination.empty());
  if (auto XDGCacheHome = sys::Process::GetEnv(XDGVariable.str())) {
    sys::path::append(Destination, *XDGCacheHome);
  } else {
    SmallString<64> PathHome;
    sys::path::home_directory(PathHome);
    sys::path::append(Destination, PathHome.str(), Default.str());
  }
}

} // namespace
