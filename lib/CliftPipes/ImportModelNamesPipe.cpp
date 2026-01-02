//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/CliftImportModel/ImportModel.h"
#include "revng/CliftPipes/CliftContainer.h"
#include "revng/CliftPipes/ImportModelNamesPipe.h"
#include "revng/Pipeline/RegisterPipe.h"

//
// Old style pipes
//

class ImportFunctionModelNamesPipe {
public:
  static constexpr auto Name = "import-function-model-names";

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
    mlir::ModuleOp Module = CliftFunctionContainer.getModule();
    const model::Binary &Binary = *revng::getModelFromContext(EC);

    for (const model::Function &Function :
         revng::getFunctionsAndCommit(EC, CliftFunctionContainer.name())) {

      mlir::clift::importNames(Function, Binary, Module);
    }
  }
};

static pipeline::RegisterPipe<ImportFunctionModelNamesPipe> X;

class ImportModelNamesPipe {
public:
  static constexpr auto Name = "import-model-names";

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
    mlir::clift::importNames(*revng::getModelFromContext(EC),
                             CliftContainer.getModule());

    EC.commitUniqueTarget(CliftContainer);
  }
};

static pipeline::RegisterPipe<ImportModelNamesPipe> Y;

//
// New style pipes
//

namespace revng::pypeline::piperuns {

void ImportFunctionModelNames::runOnCliftFunction(const model::Function
                                                    &Function,
                                                  mlir::clift::FunctionOp
                                                    MLIRFunction) {
  mlir::clift::importNames(Binary,
                           MLIRFunction->getParentOfType<mlir::ModuleOp>());
}

void ImportModelNames::run() {
  mlir::clift::importNames(Binary, TypesAndGlobals.getModule());
}

} // namespace revng::pypeline::piperuns
