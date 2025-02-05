#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/ADT/SmallString.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Signals.h"

#include "revng/Support/Debug.h"
#include "revng/Support/Error.h"

class TemporaryFile {
private:
  llvm::SmallString<32> Path;

  TemporaryFile(llvm::StringRef Path, int /* dummy */) : Path(Path.str()) {
    llvm::sys::RemoveFileOnSignal(Path);
  }

public:
  TemporaryFile(const llvm::Twine &Prefix, llvm::StringRef Suffix = "") {
    revng::cantFail(llvm::sys::fs::createTemporaryFile(Prefix, Suffix, Path));
    llvm::sys::RemoveFileOnSignal(Path);
  }

  static llvm::ErrorOr<TemporaryFile> make(const llvm::Twine &Prefix,
                                           llvm::StringRef Suffix = "") {
    using llvm::sys::fs::createTemporaryFile;
    llvm::SmallString<32> TempPath;
    int FD;
    if (auto EC = createTemporaryFile(Prefix, Suffix, FD, TempPath); EC)
      return EC;

    return TemporaryFile(TempPath, 0);
  }

  TemporaryFile(TemporaryFile &&Other) { *this = std::move(Other); }
  TemporaryFile &operator=(TemporaryFile &&Other) {
    if (not Path.empty()) {
      revng::cantFail(llvm::sys::fs::remove(Path));
      llvm::sys::DontRemoveFileOnSignal(Path);
    }

    Path = Other.Path;
    Other.Path.clear();

    return *this;
  }

  ~TemporaryFile() {
    if (not Path.empty()) {
      revng::cantFail(llvm::sys::fs::remove(Path));
      llvm::sys::DontRemoveFileOnSignal(Path);
    }
  }

public:
  TemporaryFile(const TemporaryFile &) = delete;
  TemporaryFile &operator=(const TemporaryFile &) = delete;

public:
  llvm::StringRef path() const {
    revng_assert(Path.size() > 0);
    return Path;
  }
};
