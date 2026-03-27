#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <optional>

#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Types.h"

#include "revng/Clift/Clift.h"
#include "revng/Model/Binary.h"

namespace mlir::clift {

/// Convert the specified unqualified model type to a Clift type in the
/// specified context.
///
/// \return The corresponding Clift ValueType, or null on failure.
ValueType importType(llvm::function_ref<mlir::InFlightDiagnostic()> EmitError,
                     mlir::MLIRContext &Context,
                     const model::TypeDefinition &ModelType,
                     const model::Binary &Binary);

/// Convert the specified qualified model type to a Clift type in the specified
/// context.
///
/// \return The corresponding Clift ValueType, or null on failure.
ValueType importType(llvm::function_ref<mlir::InFlightDiagnostic()> EmitError,
                     mlir::MLIRContext &Context,
                     const model::Type &ModelType,
                     const model::Binary &Binary);

/// Convert the specified model function into a clift function declaration.
///
/// \return The corresponding Clift FunctionOp, or null on failure.
FunctionOp
importFunctionDeclaration(mlir::ModuleOp Module,
                          mlir::Location DebugLocation,
                          llvm::StringRef Name,
                          llvm::StringRef Handle,
                          clift::FunctionType Prototype,
                          const model::Function::TypeOfAttributes &Attributes);

/// Convert the specified model segment into a clift variable.
///
/// \return The corresponding Clift GlobalVariableOp, or null on failure.
GlobalVariableOp importSegmentDeclaration(mlir::ModuleOp Module,
                                          mlir::Location DebugLocation,
                                          llvm::StringRef Name,
                                          llvm::StringRef Handle,
                                          clift::ValueType Type);

void importAllModelTypes(const model::Binary &Model, mlir::ModuleOp Module);

void importDescriptiveInfo(const model::Binary &Model, mlir::ModuleOp Module);

// NOTE: this overload can be dropped together with the old pipeline.
void importDescriptiveInfo(const model::Function &Function,
                           const model::Binary &Model,
                           mlir::ModuleOp Module);

} // namespace mlir::clift
