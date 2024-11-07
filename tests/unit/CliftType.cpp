//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "mlir/IR/Diagnostics.h"

#include "revng-c/mlir/Dialect/Clift/IR/Clift.h"
#include "revng-c/mlir/Dialect/Clift/Utils/ImportModel.h"

template<typename CallableType>
static auto withContext(CallableType Callable) {
  mlir::MLIRContext Context;
  Context.loadDialect<mlir::clift::CliftDialect>();

  const auto EmitError = [&]() -> mlir::InFlightDiagnostic {
    return Context.getDiagEngine().emit(mlir::UnknownLoc::get(&Context),
                                        mlir::DiagnosticSeverity::Remark);
  };

  return Callable(EmitError, Context);
}

static bool verify(const model::TypeDefinition &ModelType,
                   const model::Binary &Binary,
                   const bool Assert) {
  return withContext([&](const auto EmitError, mlir::MLIRContext &Context) {
    return static_cast<bool>(mlir::clift::importModelType(EmitError,
                                                          Context,
                                                          ModelType,
                                                          Binary));
  });
}

static bool
verify(const model::Type &ModelType, const model::Binary &Binary, bool Assert) {
  return withContext([&](const auto EmitError, mlir::MLIRContext &Context) {
    return static_cast<bool>(mlir::clift::importModelType(EmitError,
                                                          Context,
                                                          ModelType,
                                                          Binary));
  });
}

static bool verify(const model::Binary &Tree, const bool Assert) {
  return Tree.verify(Assert);
}

static bool checkSerialization(const TupleTree<model::Binary> &Tree) {
  return true;
}

#include "revng/tests/unit/ModelType.inc"
