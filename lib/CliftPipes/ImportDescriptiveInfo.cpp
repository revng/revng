//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/CliftImportModel/ImportModel.h"
#include "revng/CliftPipes/ImportDescriptiveInfo.h"

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
