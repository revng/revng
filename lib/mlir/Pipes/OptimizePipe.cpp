//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/Passes.h"

#include "revng/Pipeline/RegisterPipe.h"
#include "revng/Pipes/Kinds.h"
#include "revng/Pipes/ModelGlobal.h"
#include "revng/mlir/Dialect/Clift/Utils/Helpers.h"
#include "revng/mlir/Pipes/CliftContainer.h"

using namespace revng;
namespace clift = mlir::clift;

namespace {

class OptimizeCliftPipe {
public:
  static constexpr auto Name = "optimize-clift";

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

    // Create the PassManager and add the passes to the schedule
    mlir::ModuleOp Module = CliftContainer.getModule();
    mlir::PassManager PM(Module.getContext());

    // Add the beautify passes
    PM.addPass(mlir::createCanonicalizerPass());

    // Run the PassManager
    mlir::LogicalResult Result = PM.run(Module);
    revng_check(Result.succeeded());

    EC.commitAllFor(CliftContainer);
  }
};

static pipeline::RegisterPipe<OptimizeCliftPipe> X;

} // namespace
