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
           revng::pipes::CliftFunctionContainer &CliftContainer) {
    mlir::ModuleOp Module = CliftContainer.getModule();
    const model::Binary &Model = *revng::getModelFromContext(EC);

    for (const model::Function &Function :
         revng::getFunctionsAndCommit(EC, CliftContainer.name())) {
      // Note that this re-imports *every* global for *every* function, which
      // is really bad from the invalidation stand point.
      //
      // The proper solution would be to manually determine which functions
      // use which globals - and only update those BUT this problem is only
      // affecting the old pipeline (in the new one, every function is in
      // a separate module only containing its dependencies).
      //
      // As such, it's not worth fixing it at this point: we can live with
      // a bunch of unnecessary invalidations until we drop the old pipeline.
      mlir::clift::importNames(Function, Model, Module);
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

using IFMN = ::revng::pypeline::piperuns::ImportFunctionModelNames;
void IFMN::runOnCliftFunction(const model::Function &Function,
                              mlir::clift::FunctionOp MLIR) {
  mlir::clift::importNames(Binary, MLIR->getParentOfType<mlir::ModuleOp>());
}

void ImportModelNames::run() {
  mlir::clift::importNames(Binary, TypesAndGlobals.getModule());
}

} // namespace revng::pypeline::piperuns
