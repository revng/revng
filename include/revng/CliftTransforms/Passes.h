#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "mlir/Pass/Pass.h"

#include "revng/Clift/Clift.h"

namespace clift {

template<typename OpT>
using PassPtr = std::unique_ptr<mlir::OperationPass<OpT>>;

#define GEN_PASS_DECL
#include "revng/CliftTransforms/Passes.h.inc"

PassPtr<clift::FunctionOp> createTrivialReturnEliminationPass();

PassPtr<clift::FunctionOp> createLabelMergingPass();
PassPtr<clift::FunctionOp> createLoopDetectionPass();

PassPtr<clift::FunctionOp> createOptimizeStatementsPass();
PassPtr<clift::FunctionOp> createOptimizeExpressionsPass();

PassPtr<clift::FunctionOp> createEmitFieldAccessesPass();

PassPtr<clift::FunctionOp> createTerminalBranchComplementHoistingPass();
PassPtr<clift::FunctionOp> createVariableScopeTighteningPass();
PassPtr<clift::FunctionOp> createVariableInitializerHoistingPass();
PassPtr<clift::FunctionOp> createPromoteBreakContinuePass();

PassPtr<clift::FunctionOp> createCLegalizationPass();

PassPtr<clift::FunctionOp> createImmediateRadixDeductionPass();
PassPtr<clift::FunctionOp> createImplicitCastElisionPass();

PassPtr<mlir::ModuleOp> createVerifyCPass();
PassPtr<mlir::ModuleOp> createEmitCPass();

PassPtr<mlir::ModuleOp> createEmitTypeAndGlobalHeaderPass();
PassPtr<mlir::ModuleOp> createEmitHelperHeaderPass();

#define GEN_PASS_REGISTRATION
#include "revng/CliftTransforms/Passes.h.inc"

} // namespace clift
