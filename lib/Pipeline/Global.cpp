/// \file Global.cpp
/// A saveable object that an be serialized and deserialized from a string.

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <system_error>

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/raw_ostream.h"

#include "revng/Pipeline/Global.h"

using namespace std;
using namespace pipeline;
using namespace llvm;

Error Global::storeToDisk(const revng::FilePath &Path) const {
  auto MaybeWritableFile = Path.getWritableFile();
  if (not MaybeWritableFile)
    return MaybeWritableFile.takeError();

  auto &WritableFile = MaybeWritableFile.get();
  llvm::Error SerializeError = serialize(WritableFile.get()->os());
  if (SerializeError)
    return SerializeError;

  return WritableFile.get()->commit();
}

Error Global::loadFromDisk(const revng::FilePath &Path) {
  auto MaybeExists = Path.exists();
  if (not MaybeExists)
    return MaybeExists.takeError();

  if (not MaybeExists.get()) {
    clear();
    return llvm::Error::success();
  }

  auto MaybeBuffer = Path.getReadableFile();
  if (not MaybeBuffer) {
    return MaybeBuffer.takeError();
  }

  llvm::Error DeserializeError = deserialize(MaybeBuffer.get()->buffer());
  return DeserializeError;
}
