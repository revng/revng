//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/Pipeline/RegisterPipe.h"
#include "revng/Pipes/ModelGlobal.h"
#include "revng/mlir/Dialect/Clift/Utils/Helpers.h"
#include "revng/mlir/Dialect/Clift/Utils/LoopPromotion.h"
#include "revng/mlir/Pipes/CliftContainer.h"

using namespace revng;
namespace clift = mlir::clift;

namespace {

class LoopPromotionPipe {
public:
  static constexpr auto Name = "loop-promotion-clift";

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

#if 0
    auto R = clift::verifyAgainstModel(CliftContainer.getModule(),
                                       *revng::getModelFromContext(EC));
    revng_assert(R.succeeded());
#endif

#if 0
    for (const model::Function &Function : revng::getFunctionsAndCommit(EC, CliftContainer.name())) {

    }
#endif

    mlir::ModuleOp Module = CliftContainer.getModule();

    std::unordered_map<MetaAddress, clift::FunctionOp> Functions;
    Module->walk([&](clift::FunctionOp F) {
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

      mlir::clift::inspectLoops(It->second);
      // revng_check(R.succeeded());
    }

    // Module.dump();

    // EC.commitAllFor(CliftContainer);
  }
};

static pipeline::RegisterPipe<LoopPromotionPipe> X;

} // namespace
