//
// Copyright (c) rev.ng Srls. See LICENSE.md for details.
//

#include "llvm/ADT/StringRef.h"
#include "llvm/IR/Function.h"

#include "revng/Model/Binary.h"
#include "revng/Support/Assert.h"

bool hasIsolatedFunction(const model::Binary &Model, const std::string &FName) {

  for (const model::Function &ModelFunction : Model.Functions)
    if (FName == ModelFunction.Name)
      return true;

  return false;
}

bool hasIsolatedFunction(const model::Binary &Model, const llvm::Function *F) {
  revng_assert(F);
  return hasIsolatedFunction(Model, F->getName().str());
}
