/// \file ModelGlobal.cpp
/// \brief The model global is a wrapper around a model to be used in the
/// pipeline context

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/Pipes/ModelGlobal.h"

using namespace std;
using namespace pipeline;
using namespace revng::pipes;

const char ModelGlobal::ID = '0';

llvm::Error ModelGlobal::storeToDisk(llvm::StringRef Path) const {
  return serializeToFile(Model.getReadOnlyModel(), Path);
}

llvm::Error ModelGlobal::loadFromDisk(llvm::StringRef Path) {
  if (not llvm::sys::fs::exists(Path))
    return llvm::Error::success();

  auto MaybeModel = TupleTree<model::Binary>::fromFile(Path);
  if (not MaybeModel)
    return llvm::make_error<llvm::StringError>("Could not parse model",
                                               MaybeModel.getError());

  Model = std::move(*MaybeModel);
  return llvm::Error::success();
}
