//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/CliftImportModel/Verify.h"
#include "revng/CliftPipes/VerifyAgainstModel.h"
#include "revng/Model/Binary.h"

namespace {

void verifyImpl(mlir::ModuleOp Module, const model::Binary &Binary) {
  // If the verify logger is disabled, this pipe does nothing
  if (not ModelVerifyLogger.isEnabled())
    return;

  // This pipe reads a lot of the model in order to assert that some properties
  // of it are correct in the MLIR module, however it does not write to it in
  // any way. Because of this, we disable the tracking temporarily as to not
  // have the model paths read here count for invalidation purposes.
  DisableTracking<model::Binary> Guard(Binary);

  auto R = clift::verifyAgainstModel(Module, Binary);
  revng_assert(R.succeeded());
}

} // namespace

namespace revng::pypeline::piperuns {

void VerifyFunctionAgainstModel::runOnCliftFunction(const model::Function &,
                                                    clift::FunctionOp
                                                      Function) {
  verifyImpl(Function->getParentOfType<mlir::ModuleOp>(), Binary);
}

void VerifyAgainstModel::run() {
  verifyImpl(TypesAndGlobals.getModule(), Binary);
}

} // namespace revng::pypeline::piperuns
