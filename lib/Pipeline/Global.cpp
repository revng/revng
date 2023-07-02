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

Error Global::storeToDisk(StringRef Path) const {
  std::error_code EC;
  raw_fd_ostream OS(Path, EC, llvm::sys::fs::OF_None);
  if (EC)
    return createStringError(EC,
                             "could not write file at %s",
                             Path.str().c_str());

  return serialize(OS);
}

Error Global::loadFromDisk(StringRef Path) {
  if (not llvm::sys::fs::exists(Path)) {
    clear();
    return llvm::Error::success();
  }

  if (auto MaybeBuffer = MemoryBuffer::getFile(Path); !MaybeBuffer)
    return llvm::createStringError(MaybeBuffer.getError(),
                                   "could not read file at %s",
                                   Path.str().c_str());
  else
    return deserialize(**MaybeBuffer);
}
