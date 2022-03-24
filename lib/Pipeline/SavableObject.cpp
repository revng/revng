/// \file SavableObject.cpp
/// \brief a savable object is a objecet that an be serialized and deserialized
/// froms a string

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <system_error>

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/raw_ostream.h"

#include "revng/Pipeline/SavableObject.h"

using namespace std;
using namespace pipeline;
using namespace llvm;

Error SavableObjectBase::storeToDisk(StringRef Path) const {
  std::error_code EC;
  raw_fd_ostream OS(Path, EC, llvm::sys::fs::F_None);
  if (EC)
    return createStringError(EC,
                             "could not write file at %s",
                             Path.str().c_str());

  return serialize(OS);
}

Error SavableObjectBase::loadFromDisk(StringRef Path) {
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
