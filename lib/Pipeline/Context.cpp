/// \file Context.cpp
/// The pipeline context the place where all objects used by more that one
/// pipeline or container are stored.

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <cstdlib>

#include "llvm/ADT/StringRef.h"

#include "revng/Pipeline/AllRegistries.h"
#include "revng/Pipeline/Context.h"

using namespace pipeline;

Logger<> pipeline::ExplanationLogger("pipeline");
Logger<> pipeline::CommandLogger("commands");

Context::Context() : TheKindRegistry(Registry::registerAllKinds()) {
}

llvm::Error Context::store(const revng::DirectoryPath &Path) const {
  if (auto Error = Globals.store(Path); Error)
    return Error;

  revng::FilePath IndexPath = Path.getFile("index");
  auto MaybeWritableFile = IndexPath.getWritableFile();
  if (not MaybeWritableFile)
    return MaybeWritableFile.takeError();

  MaybeWritableFile->get()->os() << CommitIndex << "\n";
  return MaybeWritableFile->get()->commit();
}

llvm::Error Context::load(const revng::DirectoryPath &Path) {
  if (auto Error = Globals.load(Path); Error)
    return Error;

  revng::FilePath IndexPath = Path.getFile("index");

  auto MaybeExists = IndexPath.exists();
  if (not MaybeExists)
    return MaybeExists.takeError();

  if (not MaybeExists.get())
    return llvm::Error::success();

  auto MaybeReadableFile = IndexPath.getReadableFile();
  if (not MaybeReadableFile)
    return MaybeReadableFile.takeError();

  llvm::StringRef Buffer = MaybeReadableFile->get()->buffer().getBuffer();
  if (Buffer.trim().getAsInteger(10, CommitIndex)) {
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "Malformed index file");
  }

  return llvm::Error::success();
}
