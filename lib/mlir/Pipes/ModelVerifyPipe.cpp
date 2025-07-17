//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/Pipeline/RegisterPipe.h"
#include "revng/mlir/Dialect/Clift/Utils/ModelVerify.h"
#include "revng/mlir/Pipes/CliftContainer.h"

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
           revng::pipes::CliftContainer &CliftContainer) {

    auto R = clift::verifyAgainstModel(CliftContainer.getModule(),
                                       *revng::getModelFromContext(EC));
    revng_assert(R.succeeded());

    EC.commitAllFor(CliftContainer);
  }
};

static pipeline::RegisterPipe<ModelVerifyPipe> X;

} // namespace
