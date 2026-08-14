//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Signals.h"

#include "mlir/Dialect/LLVMIR/Transforms/Passes.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "mlir/Transforms/Passes.h"

#include "revng/Clift/Clift.h"
#include "revng/CliftTransforms/Passes.h"
#include "revng/Support/Assert.h"
#include "revng/Support/InitRevng.h"

using namespace llvm::cl;

static constexpr char ToolName[] = "Standalone optimizer driver\n";

static void initializeCliftDialect(mlir::MLIRContext *Context,
                                   clift::CliftDialect *Dialect) {
  auto DataModel = CDataModel::getDefaultDataModel(8);

  // Enable 128-bit integer support:
  DataModel.ExtendedIntegerSizeMask |= 128 / 8;

  Dialect->setDefaultDataModel(DataModel);
}

int main(int Argc, char *Argv[]) {
  mlir::DialectRegistry Registry;

  Registry.insert<clift::CliftDialect>();
  Registry.addExtension(initializeCliftDialect);

  mlir::registerTransformsPasses();
  clift::registerCliftPasses();

  using mlir::asMainReturnCode;
  using mlir::MlirOptMain;

  return asMainReturnCode(MlirOptMain(Argc, Argv, ToolName, Registry));
}
