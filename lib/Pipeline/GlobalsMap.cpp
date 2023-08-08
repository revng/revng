/// \file GlobalsMap.cpp

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

llvm::Error GlobalsMap::store(const revng::DirectoryPath &Path) const {
  for (const auto &Global : Map) {
    revng::FilePath Filename = Path.getFile(Global.first);
    if (auto E = Global.second->store(Filename); !!E)
      return E;
  }
  return llvm::Error::success();
}

llvm::Error GlobalsMap::load(const revng::DirectoryPath &Path) {
  for (const auto &Global : Map) {
    revng::FilePath Filename = Path.getFile(Global.first);
    if (auto E = Global.second->load(Filename); !!E)
      return E;
  }
  return llvm::Error::success();
}
