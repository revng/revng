//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/Clift/CliftDialect.h"
#include "revng/Clift/Helpers.h"
#include "revng/CliftImportModel/ImportModel.h"
#include "revng/CliftPipes/CliftContainer.h"
#include "revng/CliftPipes/ImportTypes.h"
#include "revng/Pipeline/Location.h"
#include "revng/Pipeline/RegisterPipe.h"
#include "revng/Pipes/FileContainer.h"
#include "revng/Pipes/Kinds.h"

//
// Old style pipes
//

class ImportTypesPipe {
public:
  static constexpr auto Name = "import-types";

  std::array<pipeline::ContractGroup, 1> getContract() const {
    return { pipeline::ContractGroup(revng::kinds::Binary,
                                     0,
                                     revng::kinds::CliftModule,
                                     1) };
  }

  void run(pipeline::ExecutionContext &EC,
           const revng::pipes::BinaryFileContainer &,
           revng::pipes::CliftContainer &CliftContainer) {
    mlir::MLIRContext *Context = CliftContainer.getContext();
    Context->loadDialect<clift::CliftDialect>();

    clift::importAllModelTypes(*revng::getModelFromContext(EC),
                               CliftContainer.getModule());

    EC.commitAllFor(CliftContainer);
  }
};

static pipeline::RegisterPipe<ImportTypesPipe> Y;

class ImportFunctionDeclarationsPipe {
public:
  static constexpr auto Name = "import-function-declarations";

  std::array<pipeline::ContractGroup, 1> getContract() const {
    return { pipeline::ContractGroup(revng::kinds::CliftModule, 0) };
  }

  void run(pipeline::ExecutionContext &EC,
           revng::pipes::CliftContainer &CliftContainer) {
    mlir::MLIRContext *Context = CliftContainer.getContext();
    Context->loadDialect<clift::CliftDialect>();

    clift::importAllModelFunctionDeclarations(*revng::getModelFromContext(EC),
                                              CliftContainer.getModule());

    EC.commitAllFor(CliftContainer);
  }
};

static pipeline::RegisterPipe<ImportFunctionDeclarationsPipe> IFD;

class ImportSegmentDeclarationsPipe {
public:
  static constexpr auto Name = "import-segment-declarations";

  std::array<pipeline::ContractGroup, 1> getContract() const {
    return { pipeline::ContractGroup(revng::kinds::CliftModule, 0) };
  }

  void run(pipeline::ExecutionContext &EC,
           revng::pipes::CliftContainer &CliftContainer) {
    mlir::MLIRContext *Context = CliftContainer.getContext();
    Context->loadDialect<clift::CliftDialect>();

    clift::importAllModelSegmentDeclarations(*revng::getModelFromContext(EC),
                                             CliftContainer.getModule());

    EC.commitAllFor(CliftContainer);
  }
};

static pipeline::RegisterPipe<ImportSegmentDeclarationsPipe> ISD;

//
// New style pipes
//

namespace revng::pypeline::piperuns {

void ImportTypes::run() {
  clift::importAllModelTypes(Binary, Output.getModule());
}

void ImportFunctionDeclarations::run() {
  clift::importAllModelFunctionDeclarations(Binary, Module.getModule());
}

} // namespace revng::pypeline::piperuns
