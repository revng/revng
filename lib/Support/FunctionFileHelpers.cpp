//
// Copyright rev.ng Srls. See LICENSE.md for details.
//

#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"

#include "revng/Support/Assert.h"

#include "revng-c/Support/FunctionFileHelpers.h"

using namespace llvm;

std::unique_ptr<llvm::raw_fd_ostream>
openFunctionFile(const StringRef DirectoryPath,
                 const StringRef FunctionName,
                 const StringRef Suffix) {

  std::error_code Error;
  SmallString<32> FilePath = DirectoryPath;

  if (FilePath.empty())
    if ((Error = llvm::sys::fs::current_path(FilePath)))
      revng_abort(Error.message().c_str());

  if ((Error = llvm::sys::fs::make_absolute(FilePath)))
    revng_abort(Error.message().c_str());

  if ((Error = llvm::sys::fs::create_directories(FilePath)))
    revng_abort(Error.message().c_str());

  llvm::sys::path::append(FilePath, FunctionName + Suffix);
  auto FileOStream = std::make_unique<llvm::raw_fd_ostream>(FilePath, Error);
  if (Error) {
    FileOStream.reset();
    revng_abort(Error.message().c_str());
  }

  return FileOStream;
}
