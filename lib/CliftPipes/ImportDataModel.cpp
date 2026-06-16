//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/ABI/Definition.h"
#include "revng/CliftPipes/CliftContainer.h"
#include "revng/CliftPipes/ImportDataModel.h"
#include "revng/Pipeline/RegisterPipe.h"

static void importDataModel(mlir::ModuleOp Module, const model::Binary &Model) {
  clift::setDataModel(Module, abi::getDataModel(Model));
}

//
// Old style pipes
//

struct ImportFunctionDataModelPipe {
  static constexpr auto Name = "import-function-data-model";

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
           revng::pipes::CliftFunctionContainer &CliftContainer) {
    importDataModel(CliftContainer.getModule(),
                    *revng::getModelFromContext(EC));

    EC.commitAllFor(CliftContainer);
  }
};

static pipeline::RegisterPipe<ImportFunctionDataModelPipe> X;

struct ImportDataModelPipe {
  static constexpr auto Name = "import-data-model";

  std::array<pipeline::ContractGroup, 1> getContract() const {
    using namespace pipeline;
    using namespace revng::kinds;

    return { ContractGroup({ Contract(CliftModule,
                                      0,
                                      CliftModule,
                                      0,
                                      InputPreservation::Preserve) }) };
  }

  void run(pipeline::ExecutionContext &EC,
           revng::pipes::CliftContainer &CliftContainer) {
    importDataModel(CliftContainer.getModule(),
                    *revng::getModelFromContext(EC));

    EC.commitUniqueTarget(CliftContainer);
  }
};

static pipeline::RegisterPipe<ImportDataModelPipe> Y;

//
// New style pipes
//

namespace revng::pypeline::piperuns {

using IFDM = ImportFunctionDataModel;
void IFDM::runOnCliftFunction(const model::Function &Function,
                              clift::FunctionOp MLIR) {
  importDataModel(MLIR->getParentOfType<mlir::ModuleOp>(), Binary);
}

void ImportDataModel::run() {
  importDataModel(TypesAndGlobals.getModule(), Binary);
}

} // namespace revng::pypeline::piperuns
