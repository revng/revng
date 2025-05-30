//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "revng/mlir/Dialect/Clift/IR/CliftOpHelpers.h"
#include "revng/mlir/Dialect/Clift/IR/CliftOps.h"
#include "revng/mlir/Dialect/Clift/Transforms/Passes.h"

namespace mlir {
namespace clift {
#define GEN_PASS_DEF_CLIFTEXPRESSIONROOTSIMPLIFICATION
#include "revng/mlir/Dialect/Clift/Transforms/Passes.h.inc"
} // namespace clift
} // namespace mlir

namespace clift = mlir::clift;

namespace {

static void simplifyExpressionRoot(clift::YieldOp Yield) {
  
}

template<typename T>
using PassBase = clift::impl::CliftExpressionRootSimplificationBase<T>;

struct ExpressionRootSimplificationPass
  : PassBase<ExpressionRootSimplificationPass> {

  void runOnOperation() override {
    getOperation()->walk([](clift::StatementOpInterface Statement) {
      mlir::Operation *Op = Statement.getOperation();

      if (auto S = mlir::dyn_cast<clift::ExpressionStatementOp>(Op)) {
        simplifyExpressionRoot(clift::getYieldOp(S.getExpression()));
      } else if (auto S = mlir::dyn_cast<clift::BranchOpInterface>(Op)) {
        // WIP: Add new ControlFlowOpInterface and use that instead.
        simplifyExpressionRoot(clift::getYieldOp(S.getBranchCondition()));
      }
    });
  }
};

} // namespace

clift::PassPtr<clift::FunctionOp>
clift::createExpressionRootSimplificationPass() {
  return std::make_unique<ExpressionRootSimplificationPass>();
}
