//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "mlir/IR/MLIRContext.h"

#include "revng/Clift/CliftDialect.h"
#include "revng/Clift/CliftTypes.h"
#include "revng/Clift/Helpers.h"
#include "revng/CliftImportModel/ImportModel.h"
#include "revng/CliftPipes/CliftContainer.h"
#include "revng/CliftPipes/ImportCliftPipes.h"
#include "revng/Model/Segment.h"
#include "revng/Pipeline/Location.h"
#include "revng/Pipeline/RegisterPipe.h"
#include "revng/Pipes/FileContainer.h"
#include "revng/Pipes/Kinds.h"
#include "revng/Pipes/Ranks.h"

namespace clift = mlir::clift;

//
// Old style pipes
//

class ImportCliftTypesPipe {
public:
  static constexpr auto Name = "import-clift-types";

  std::array<pipeline::ContractGroup, 1> getContract() const {
    return { pipeline::ContractGroup(revng::kinds::Binary,
                                     0,
                                     revng::kinds::CliftModule,
                                     1) };
  }

  void run(pipeline::ExecutionContext &EC,
           const revng::pipes::BinaryFileContainer &,
           revng::pipes::CliftContainer &CliftContainer) {
    mlir::clift::importAllModelTypes(*revng::getModelFromContext(EC),
                                     CliftContainer.getModule());

    EC.commitUniqueTarget(CliftContainer);
  }
};

static pipeline::RegisterPipe<ImportCliftTypesPipe> Y;

class ImportFunctionDeclarations {
public:
  static constexpr auto Name = "import-clift-function-declarations";

  std::array<pipeline::ContractGroup, 1> getContract() const {
    return { pipeline::ContractGroup(revng::kinds::CliftModule,
                                     0,
                                     pipeline::InputPreservation::Preserve) };
  }

  void run(pipeline::ExecutionContext &EC,
           revng::pipes::CliftContainer &CliftContainer) {
    const model::Binary &Binary = *revng::getModelFromContext(EC);
    mlir::clift::importAllModelFunctionDeclarations(Binary,
                                                    CliftContainer.getModule());

    EC.commitUniqueTarget(CliftContainer);
  }
};

static pipeline::RegisterPipe<ImportFunctionDeclarations> Z;

class ImportSegmentDeclarations {
public:
  static constexpr auto Name = "import-clift-segment-declarations";

  std::array<pipeline::ContractGroup, 1> getContract() const {
    return { pipeline::ContractGroup(revng::kinds::CliftModule,
                                     0,
                                     pipeline::InputPreservation::Preserve) };
  }

  void run(pipeline::ExecutionContext &EC,
           revng::pipes::CliftContainer &CliftContainer) {
    const model::Binary &Binary = *revng::getModelFromContext(EC);
    mlir::clift::importAllModelSegmentDeclarations(Binary,
                                                   CliftContainer.getModule());

    EC.commitUniqueTarget(CliftContainer);
  }
};

static pipeline::RegisterPipe<ImportSegmentDeclarations> A;

//
// New style pipes
//

namespace revng::pypeline::piperuns {

void ImportCliftTypes::run() {
  mlir::clift::importAllModelTypes(Binary, Output.getModule());
}

void ImportCliftFunctionDeclarations::run() {
  mlir::clift::importAllModelFunctionDeclarations(Binary, Module.getModule());
}

} // namespace revng::pypeline::piperuns
