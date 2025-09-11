/// \file Path.cpp

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/Support/FileSystem.h"

#include "revng/Storage/Path.h"
#include "revng/Storage/StorageClient.h"

namespace {

// TODO: unix-specific, will not work on Windows
revng::StorageClient LocalClient("/");

std::string normalizeLocalPath(llvm::StringRef Path) {
  llvm::SmallString<256> PathCopy(Path);
  revng::cantFail(llvm::sys::fs::make_absolute(PathCopy));
  llvm::sys::path::remove_dots(PathCopy, true);
  llvm::StringRef Result(PathCopy.substr(1)); // Remove leading '/'
  return Result.str();
}

} // namespace

namespace revng {

DirectoryPath DirectoryPath::fromLocalStorage(llvm::StringRef Path) {
  return revng::DirectoryPath{ &LocalClient, normalizeLocalPath(Path) };
}

FilePath FilePath::fromLocalStorage(llvm::StringRef Path) {
  return revng::FilePath{ &LocalClient, normalizeLocalPath(Path) };
}

} // namespace revng
