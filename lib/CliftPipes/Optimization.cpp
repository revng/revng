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

namespace {

template<typename BaseT>
struct OptimizationPipe : BaseT {
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
           pipes::CliftFunctionContainer &CliftFunctionContainer) {
    mlir::PassManager PM(CliftFunctionContainer.getContext(),
                         clift::FunctionOp::getOperationName());

    BaseT::configurePassManager(PM);

    mlir::ModuleOp Module = CliftFunctionContainer.getModule();

    std::unordered_map<MetaAddress, clift::FunctionOp> Functions;
    Module->walk([&Functions](clift::FunctionOp F) {
      MetaAddress MA = getMetaAddress(F);
      if (MA.isValid()) {
        auto [Iterator, Inserted] = Functions.try_emplace(MA, F);
        revng_assert(Inserted);
      }
    });

    for (const model::Function &Function :
         getFunctionsAndCommit(EC, CliftFunctionContainer.name())) {
      auto It = Functions.find(Function.Entry());
      revng_check(It != Functions.end()
                  and "Requested Clift function not found");

      mlir::LogicalResult R = PM.run(It->second);
      revng_check(R.succeeded());
    }
  }
};

struct GenericOptimizationPipe {
  static constexpr auto Name = "clift-generic-optimization";

  static void configurePassManager(mlir::PassManager &PM) {
    // Eliminating trivial returns during statement rewriting (where other
    // trivial jumps are eliminated) is difficult, because the triviality of a
    // jump is dependent on its context, and if the rewrite targets the context
    // (a nesting region, as is the case for trivial jump elimination), unlike
    // for label targets, there is no easy and cheap way to iterate over the set
    // of returns in a function. For this reason, and because we expect the
    // triviality of returns not be changed by other transforms, we eliminate
    // them early in a separate pass. If in the future it turns out that there
    // are trivial returns which we cannot eliminate without first eliminating
    // other jumps, that is easily solved by replacing this pass with one which
    // converts valueless returns into gotos targeting a label at the end of the
    // function.
    PM.addPass(clift::createTrivialReturnEliminationPass());

    // Labels are merged before loop detection to avoid any issues with multiple
    // labels at the end of, or following a loop.
    PM.addPass(clift::createLabelMergingPass());
    PM.addPass(clift::createLoopDetectionPass());

    PM.addPass(clift::createOptimizeStatementsPass());
    PM.addPass(clift::createLabelMergingPass());
    PM.addPass(clift::createOptimizeExpressionsPass());

    PM.addPass(clift::createEmitFieldAccessesPass());
    PM.addPass(mlir::createCanonicalizerPass());
    PM.addPass(clift::createOptimizeExpressionsPass());

    PM.addPass(clift::createTerminalBranchComplementHoistingPass());
  }
};
static pipeline::RegisterPipe<OptimizationPipe<GenericOptimizationPipe>> X;

struct TargetOptimizationPipe {
  static constexpr auto Name = "clift-target-optimization";

  static void configurePassManager(mlir::PassManager &PM) {
    PM.addPass(clift::createCLegalizationPass(TargetCImplementation::Default));
    PM.addPass(clift::createImmediateRadixDeductionPass());
  }
};
static pipeline::RegisterPipe<OptimizationPipe<TargetOptimizationPipe>> Y;

} // namespace
