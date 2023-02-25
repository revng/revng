/// \file GlobalsMap.cpp
/// \brief GlobalsMap methods implementations

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <system_error>

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"

#include "revng/Pipeline/GlobalsMap.h"

using namespace std;
using namespace pipeline;
using namespace llvm;

llvm::Error GlobalsMap::storeToDisk(llvm::StringRef Path) const {
  for (const auto &Global : Map) {
    llvm::SmallString<128> Filename;
    llvm::sys::path::append(Filename, Path, Global.first());
    if (auto E = Global.second->storeToDisk(Filename); !!E)
      return E;
  }
  return llvm::Error::success();
}

llvm::Error GlobalsMap::loadFromDisk(llvm::StringRef Path) {
  for (const auto &Global : Map) {
    llvm::SmallString<128> Filename;
    llvm::sys::path::append(Filename, Path, Global.first());
    if (auto E = Global.second->loadFromDisk(Filename); !!E)
      return E;
  }
  return llvm::Error::success();
}
