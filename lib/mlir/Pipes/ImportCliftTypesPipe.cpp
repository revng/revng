//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/FunctionInterfaces.h"

#include "revng/EarlyFunctionAnalysis/CFGStringMap.h"
#include "revng/EarlyFunctionAnalysis/ControlFlowGraphCache.h"
#include "revng/Pipeline/Location.h"
#include "revng/Pipeline/RegisterPipe.h"
#include "revng/Pipes/Kinds.h"
#include "revng/mlir/Dialect/Clift/IR/Clift.h"
#include "revng/mlir/Dialect/Clift/IR/CliftOps.h"
#include "revng/mlir/Dialect/Clift/Utils/Helpers.h"
#include "revng/mlir/Dialect/Clift/Utils/ImportModel.h"
#include "revng/mlir/Pipes/MLIRContainer.h"

using namespace mlir::clift;

static void importAllModelTypes(const model::Binary &Model,
                                mlir::ModuleOp Module) {
  mlir::MLIRContext *const Context = Module->getContext();
  Context->loadDialect<CliftDialect>();

  const auto EmitError = [&]() -> mlir::InFlightDiagnostic {
    return Context->getDiagEngine().emit(mlir::UnknownLoc::get(Context),
                                         mlir::DiagnosticSeverity::Error);
  };

  mlir::OpBuilder Builder(Module.getRegion());
  for (const auto &ModelType : Model.TypeDefinitions()) {
    auto CliftType = importModelType(EmitError, *Context, *ModelType, Model);
    Builder.create<UndefOp>(mlir::UnknownLoc::get(Context), CliftType);
  }
}

class ImportAllCliftTypesPipe {
public:
  static constexpr auto Name = "import-all-clift-types";

  std::array<pipeline::ContractGroup, 1> getContract() const {
    using namespace pipeline;
    using namespace revng::kinds;

    return { ContractGroup({ Contract(CFG,
                                      0,
                                      MLIRFunctionKind,
                                      1,
                                      InputPreservation::Preserve),
                             Contract(MLIRFunctionKind,
                                      1,
                                      MLIRFunctionKind,
                                      1,
                                      InputPreservation::Preserve) }) };
  }

  void run(pipeline::ExecutionContext &EC,
           const revng::pipes::CFGMap &CFGMap,
           revng::pipes::MLIRContainer &MLIRContainer) {
    importAllModelTypes(*revng::getModelFromContext(EC),
                        MLIRContainer.getModule());

    EC.commitAllFor(MLIRContainer);
  }
};

static pipeline::RegisterPipe<ImportAllCliftTypesPipe> Y;
