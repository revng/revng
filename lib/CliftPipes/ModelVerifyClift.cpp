//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/CliftImportModel/ModelVerify.h"
#include "revng/CliftPipes/CliftContainer.h"
#include "revng/CliftPipes/ModelVerifyClift.h"
#include "revng/Pipeline/RegisterPipe.h"

namespace clift = mlir::clift;

namespace {

class ModelVerifyPipe {
public:
  static constexpr auto Name = "model-verify-clift";

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

static pipeline::RegisterPipe<ModelVerifyPipe> X;

} // namespace

namespace revng::pypeline::piperuns {

void ModelVerifyClift::runOnCliftFunction(const model::Function &Function,
                                          mlir::clift::FunctionOp
                                            MLIRFunction) {
  // If the verify logger is disabled, this pipe does nothing
  if (not ModelVerifyLogger.isEnabled())
    return;

  // This pipe reads a lot of the model in order to assert that some properties
  // of it are correct in the MLIR module, however it does not write to it in
  // any way. Because of this, we disable the tracking temporarily as to not
  // have the model paths read here count for invalidation purposes.
  DisableTracking<model::Binary> Guard(Binary);
  mlir::ModuleOp Module = MLIRFunction->getParentOfType<mlir::ModuleOp>();
  auto R = clift::verifyAgainstModel(Module, Binary);
  revng_assert(R.succeeded());
}

} // namespace revng::pypeline::piperuns
