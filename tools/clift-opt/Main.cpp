//
// Copyright (c) rev.ng Labs Srl. See LICENSE.md for details.
//

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Signals.h"

#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

#include "revng/Support/Assert.h"
#include "revng/Support/InitRevng.h"

#include "revng-c/mlir/Dialect/Clift/IR/Clift.h"

using namespace llvm::cl;

int main(int Argc, char *Argv[]) {

  mlir::DialectRegistry Registry;
  mlir::registerAllDialects(Registry);

  mlir::registerAllPasses();

  Registry.insert<mlir::clift::CliftDialect>();

  using mlir::asMainReturnCode;
  using mlir::MlirOptMain;
  std::string ToolName = "Standalone optimizer driver\n";

  return asMainReturnCode(MlirOptMain(Argc, Argv, ToolName, Registry));

  return 0;
}
