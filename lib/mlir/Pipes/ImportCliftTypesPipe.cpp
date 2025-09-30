//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/Pipeline/Location.h"
#include "revng/Pipeline/RegisterPipe.h"
#include "revng/Pipes/Kinds.h"
#include "revng/mlir/Dialect/Clift/IR/Clift.h"
#include "revng/mlir/Dialect/Clift/Utils/Helpers.h"
#include "revng/mlir/Dialect/Clift/Utils/ImportModel.h"
#include "revng/mlir/Pipes/CliftContainer.h"

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
  for (size_t I = 0; const auto &ModelType : Model.TypeDefinitions()) {
    auto CliftType = clift::importModelType(EmitError,
                                            *Context,
                                            *ModelType,
                                            Model);

    TypeAttrs.push_back(mlir::TypeAttr::get(CliftType));
  }

  Module->setAttr("clift.test", mlir::ArrayAttr::get(Context, TypeAttrs));
}

class ImportCliftTypesPipe {
public:
  static constexpr auto Name = "import-clift-types";

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
    importModelTypes(*revng::getModelFromContext(EC),
                     CliftContainer.getModule());

    EC.commitAllFor(CliftContainer);
  }
};

static pipeline::RegisterPipe<ImportCliftTypesPipe> Y;
