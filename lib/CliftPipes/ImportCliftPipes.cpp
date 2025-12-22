//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/Clift/CliftDialect.h"
#include "revng/Clift/Helpers.h"
#include "revng/CliftImportModel/ImportModel.h"
#include "revng/CliftPipes/CliftContainer.h"
#include "revng/Pipeline/Location.h"
#include "revng/Pipeline/RegisterPipe.h"
#include "revng/Pipes/FileContainer.h"
#include "revng/Pipes/Kinds.h"

namespace clift = mlir::clift;

static void importModelTypes(const model::Binary &Model,
                             mlir::ModuleOp Module) {
  mlir::MLIRContext *Context = Module->getContext();
  Context->loadDialect<clift::CliftDialect>();

  mlir::Location Loc = mlir::UnknownLoc::get(Context);
  auto EmitError = [&]() -> mlir::InFlightDiagnostic {
    return Context->getDiagEngine().emit(Loc, mlir::DiagnosticSeverity::Error);
  };

  llvm::SmallVector<mlir::Attribute> TypeAttrs;
  for (const auto &ModelType : Model.TypeDefinitions()) {
    auto CliftType = clift::importModelType(EmitError,
                                            *Context,
                                            *ModelType,
                                            Model);

    TypeAttrs.push_back(mlir::TypeAttr::get(CliftType));
  }

  Module->setAttr("clift.types", mlir::ArrayAttr::get(Context, TypeAttrs));
}

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
    importModelTypes(*revng::getModelFromContext(EC),
                     CliftContainer.getModule());

    EC.commitUniqueTarget(CliftContainer);
  }
};

static pipeline::RegisterPipe<ImportCliftTypesPipe> Y;
