//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/Model/VerifyHelper.h"
#include "revng/Pipes/ModelGlobal.h"

namespace pipeline {
template<>
llvm::Error TupleTreeGlobal<model::Binary>::verify() const {
  model::VerifyHelper Helper;
  if (!Value->verify(Helper)) {
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "Verify failed");
  } else {
    return llvm::Error::success();
  }
}
} // namespace pipeline
