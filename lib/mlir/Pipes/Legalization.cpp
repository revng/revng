//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <unordered_map>

#include "revng/Pipeline/RegisterPipe.h"
#include "revng/Pipes/Kinds.h"
#include "revng/mlir/Dialect/Clift/Utils/Helpers.h"
#include "revng/mlir/Dialect/Clift/Utils/Legalization.h"
#include "revng/mlir/Pipes/CliftContainer.h"

using namespace revng;
namespace clift = mlir::clift;

namespace {

class CBackendPipe {
public:
  static constexpr auto Name = "clift-legalization";

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
    const auto &Target = clift::TargetCImplementation::Default;

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

      mlir::LogicalResult R = legalizeForC(It->second, Target);
      revng_check(R.succeeded());
    }
  }
};

static pipeline::RegisterPipe<CBackendPipe> X;

} // namespace
