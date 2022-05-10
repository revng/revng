/// \file ModelGlobal.cpp
/// \brief The model global is a wrapper around a model to be used in the
/// pipeline context

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/Support/Error.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"

#include "revng/Pipes/ModelGlobal.h"

using namespace std;
using namespace pipeline;
using namespace ::revng::pipes;

const char ModelGlobal::ID = '0';

llvm::Error ModelGlobal::serialize(llvm::raw_ostream &OS) const {
  Model.serialize(OS);
  return llvm::Error::success();
}

llvm::Error ModelGlobal::deserialize(const llvm::MemoryBuffer &Buffer) {
  if (auto MaybeBin = TupleTree<model::Binary>::deserialize(Buffer.getBuffer());
      !MaybeBin)
    return llvm::errorCodeToError(MaybeBin.getError());
  else
    Model = *MaybeBin;

  return llvm::Error::success();
}
