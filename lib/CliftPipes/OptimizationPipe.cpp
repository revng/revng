//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <unordered_map>

#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

#include "revng/Clift/Helpers.h"
#include "revng/CliftPipes/CliftContainer.h"
#include "revng/CliftTransforms/Passes.h"
#include "revng/Pipeline/RegisterPipe.h"
#include "revng/Pipes/Kinds.h"

using namespace revng;
namespace clift = mlir::clift;

namespace {

class OptimizationPipe {
public:
  static constexpr auto Name = "clift-optimization";

  std::array<pipeline::ContractGroup, 1> getContract() const {
    using namespace pipeline;
    using namespace kinds;

    return { ContractGroup({ Contract(CliftFunction,
                                      0,
                                      CliftFunction,
                                      0,
                                      InputPreservation::Preserve) }) };
  }

  void run(pipeline::ExecutionContext &EC,
           pipes::CliftContainer &CliftContainer) {
    mlir::PassManager PM(CliftContainer.getContext(),
                         clift::FunctionOp::getOperationName());

    PM.addPass(mlir::createCanonicalizerPass());
    PM.addPass(clift::createLoopDetectionPass());
    PM.addPass(clift::createCLegalizationPass(TargetCImplementation::Default));

    mlir::ModuleOp Module = CliftContainer.getModule();

    std::unordered_map<MetaAddress, clift::FunctionOp> Functions;
    Module->walk([&Functions](clift::FunctionOp F) {
      MetaAddress MA = getMetaAddress(F);
      if (MA.isValid()) {
        auto [Iterator, Inserted] = Functions.try_emplace(MA, F);
        revng_assert(Inserted);
      }
    });

    for (const model::Function &Function :
         getFunctionsAndCommit(EC, CliftContainer.name())) {
      auto It = Functions.find(Function.Entry());
      revng_check(It != Functions.end()
                  and "Requested Clift function not found");

      mlir::LogicalResult R = PM.run(It->second);
      revng_check(R.succeeded());
    }
  }
};

static pipeline::RegisterPipe<OptimizationPipe> X;

} // namespace
