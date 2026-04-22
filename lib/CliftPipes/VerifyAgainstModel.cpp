//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/CliftImportModel/Verify.h"
#include "revng/CliftPipes/CliftContainer.h"
#include "revng/CliftPipes/VerifyAgainstModel.h"
#include "revng/Model/Binary.h"
#include "revng/Pipeline/RegisterPipe.h"

namespace clift = mlir::clift;

namespace {

class VerifyAgainstModelPipe {
public:
  static constexpr auto Name = "verify-against-model";

  std::array<pipeline::ContractGroup, 1> getContract() const {
    using namespace pipeline;
    using namespace revng::kinds;

    return { ContractGroup({ Contract(CliftFunction,
                                      0,
                                      CliftFunction,
                                      0,
                                      InputPreservation::Preserve) }) };
  }

  void run(pipeline::ExecutionContext &EC,
           revng::pipes::CliftFunctionContainer &CliftFunctionContainer) {

    auto R = clift::verifyAgainstModel(CliftFunctionContainer.getModule(),
                                       *revng::getModelFromContext(EC));
    revng_assert(R.succeeded());

    EC.commitAllFor(CliftFunctionContainer);
  }
};

static pipeline::RegisterPipe<VerifyAgainstModelPipe> X;

} // namespace

//
// New pipeline logic starts here.
//

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
                                                    mlir::clift::FunctionOp
                                                      Function) {
  verifyImpl(Function->getParentOfType<mlir::ModuleOp>(), Binary);
}

void VerifyAgainstModel::run() {
  verifyImpl(TypesAndGlobals.getModule(), Binary);
}

} // namespace revng::pypeline::piperuns
