//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/CliftPipes/ImportDataModel.h"
#include "revng/Model/ABI/Definition.h"

static void importDataModel(mlir::ModuleOp Module, const model::Binary &Model) {
  clift::setDataModel(Module, Model.targetDataModel());
}

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
