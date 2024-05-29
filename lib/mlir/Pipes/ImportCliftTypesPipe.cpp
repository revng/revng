//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/Pipeline/Location.h"
#include "revng/Pipeline/RegisterPipe.h"

#include "revng-c/Pipes/Kinds.h"
#include "revng-c/mlir/Dialect/Clift/Utils/ImportReachableModelTypes.h"
#include "revng-c/mlir/Pipes/MLIRContainer.h"

namespace {

class ImportCliftTypesPipe {
public:
  static constexpr auto Name = "import-clift-types";

  std::array<pipeline::ContractGroup, 1> getContract() const {
    using namespace pipeline;
    using namespace revng::kinds;

    return { ContractGroup({ Contract(MLIRFunctionKind,
                                      0,
                                      MLIRFunctionKind,
                                      0,
                                      InputPreservation::Preserve) }) };
  }

  llvm::Error checkPrecondition(const pipeline::Context &Ctx) const {
    return llvm::Error::success();
  }

  void run(const pipeline::ExecutionContext &Ctx,
           revng::pipes::MLIRContainer &MLIRContainer) {
    mlir::clift::importReachableModelTypes(MLIRContainer.getModule(),
                                           *revng::getModelFromContext(Ctx));
  }
};

static pipeline::RegisterPipe<ImportCliftTypesPipe> X;
} // namespace
