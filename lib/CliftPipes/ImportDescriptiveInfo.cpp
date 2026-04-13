//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/CliftImportModel/ImportModel.h"
#include "revng/CliftPipes/CliftContainer.h"
#include "revng/CliftPipes/ImportDescriptiveInfo.h"
#include "revng/Pipeline/RegisterPipe.h"

//
// Old style pipes
//

class ImportDescriptiveFunctionInfoPipe {
public:
  static constexpr auto Name = "import-descriptive-function-info";

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
      clift::importDescriptiveInfo(Function, Model, Module);
    }
  }
};

static pipeline::RegisterPipe<ImportDescriptiveFunctionInfoPipe> X;

class ImportDescriptiveInfoPipe {
public:
  static constexpr auto Name = "import-descriptive-info";

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
    clift::importDescriptiveInfo(*revng::getModelFromContext(EC),
                                 CliftContainer.getModule());

    EC.commitUniqueTarget(CliftContainer);
  }
};

static pipeline::RegisterPipe<ImportDescriptiveInfoPipe> Y;

//
// New style pipes
//

namespace revng::pypeline::piperuns {

using IFMN = ImportDescriptiveFunctionInfo;
void IFMN::runOnCliftFunction(const model::Function &Function,
                              clift::FunctionOp MLIR) {
  clift::importDescriptiveInfo(Binary, MLIR->getParentOfType<mlir::ModuleOp>());
}

void ImportDescriptiveInfo::run() {
  clift::importDescriptiveInfo(Binary, TypesAndGlobals.getModule());
}

} // namespace revng::pypeline::piperuns
