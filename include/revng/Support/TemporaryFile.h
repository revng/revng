#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/ADT/SmallString.h"
#include "llvm/Support/FileSystem.h"

#include "revng/Support/Debug.h"

class TemporaryFile {
private:
  llvm::SmallString<32> Path;

public:
  TemporaryFile(const llvm::Twine &Prefix, llvm::StringRef Suffix = "") {
    cantFail(llvm::sys::fs::createTemporaryFile(Prefix, Suffix, Path));
  }

  TemporaryFile(TemporaryFile &&Other) { *this = std::move(Other); }
  TemporaryFile &operator=(TemporaryFile &&Other) {
    Path = Other.Path;
    Other.Path.clear();
    return *this;
  }

  ~TemporaryFile() { cantFail(llvm::sys::fs::remove(Path)); }

public:
  TemporaryFile(const TemporaryFile &) = delete;
  TemporaryFile &operator=(const TemporaryFile &) = delete;

public:
  llvm::StringRef path() const {
    revng_assert(Path.size() > 0);
    return Path;
  }

private:
  static void cantFail(std::error_code EC) { revng_assert(!EC); }
};
