/// \file Path.cpp

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/Storage/Path.h"
#include "revng/Storage/StorageClient.h"

#include "LocalStorageClient.h"
#include "StdStorageClient.h"

namespace {

// TODO: unix-specific, will not work on Windows
revng::LocalStorageClient LocalClient("/");
revng::StdinStorageClient StdinClient;
revng::StdoutStorageClient StdoutClient;

} // namespace

namespace revng {

DirectoryPath DirectoryPath::fromLocalStorage(llvm::StringRef Path) {
  llvm::SmallString<256> PathCopy(Path);
  revng_assert(!llvm::sys::fs::make_absolute(PathCopy));
  llvm::sys::path::remove_dots(PathCopy);
  llvm::StringRef NewPath(PathCopy.substr(1)); // Remove leading '/'
  return revng::DirectoryPath{ &LocalClient, NewPath };
}

FilePath FilePath::stdin() {
  return revng::FilePath{ &StdinClient, "" };
}

FilePath FilePath::stdout() {
  return revng::FilePath{ &StdoutClient, "" };
}

FilePath FilePath::fromLocalStorage(llvm::StringRef Path) {
  llvm::SmallString<256> PathCopy(Path);
  revng_assert(!llvm::sys::fs::make_absolute(PathCopy));
  llvm::sys::path::remove_dots(PathCopy);
  llvm::StringRef NewPath(PathCopy.substr(1)); // Remove leading '/'
  return revng::FilePath{ &LocalClient, NewPath };
}

} // namespace revng
