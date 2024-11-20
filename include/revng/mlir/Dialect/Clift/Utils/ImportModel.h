#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <optional>

#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Types.h"

#include "revng/Model/Binary.h"
#include "revng/mlir/Dialect/Clift/IR/Clift.h"

namespace mlir::clift {

/// Convert the specified unqualified model type to a Clift type in the
/// specified context.
///
/// \return The corresponding Clift ValueType, or null on failure.
ValueType
importModelType(llvm::function_ref<mlir::InFlightDiagnostic()> EmitError,
                mlir::MLIRContext &Context,
                const model::TypeDefinition &ModelType,
                const model::Binary &Binary);

/// Convert the specified qualified model type to a Clift type in the specified
/// context.
///
/// \return The corresponding Clift ValueType, or null on failure.
ValueType
importModelType(llvm::function_ref<mlir::InFlightDiagnostic()> EmitError,
                mlir::MLIRContext &Context,
                const model::Type &ModelType,
                const model::Binary &Binary);

} // namespace mlir::clift
