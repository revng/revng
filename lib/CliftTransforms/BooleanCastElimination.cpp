//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "revng/Clift/Clift.h"
#include "revng/Clift/CliftOpHelpers.h"
#include "revng/CliftTransforms/Passes.h"

namespace clift {
#define GEN_PASS_DEF_CLIFTBOOLEANCASTELIMINATION
#include "revng/CliftTransforms/Passes.h.inc"
} // namespace clift

using namespace clift;

namespace {
namespace boolean_cast_elimination {

static bool hasIntType(mlir::Value Value) {
  if (auto I = clift::unwrapped_dyn_cast<IntegerType>(Value.getType())) {
    return I.getKind() == IntegerKind::Signed
           and I.getSize() == getDataModel(Value).getIntSize();
  }
  return false;
}

static mlir::Value makeZeroImmediate(mlir::PatternRewriter &Rewriter,
                                     mlir::Operation *Test,
                                     mlir::Type Type) {
  auto IntSize = getDataModel(Test).getIntSize();
  auto IntType = IntegerType::get(Rewriter.getContext(),
                                  IntegerKind::Signed,
                                  IntSize);

  mlir::Value Value = Rewriter.create<ImmediateOp>(Test->getLoc(),
                                                   IntType,
                                                   llvm::APInt(IntSize * 8, 0));

  if (not equivalent(IntType, Type)) {
    Value = Rewriter.create<ImplicitCastOp>(Test->getLoc(),
                                            removeConst(Type),
                                            Value);
  }

  return Value;
}

#include "revng/CliftTransforms/BooleanCastElimination.h.inc"

} // namespace boolean_cast_elimination

struct BooleanCastEliminationPass
  : impl::CliftBooleanCastEliminationBase<BooleanCastEliminationPass> {

  mlir::FrozenRewritePatternSet Patterns;

  mlir::LogicalResult initialize(mlir::MLIRContext *Context) override {
    mlir::RewritePatternSet Set(Context);

    boolean_cast_elimination::populateWithGenerated(Set);
    Patterns = mlir::FrozenRewritePatternSet(std::move(Set));

    return mlir::success();
  }

  void runOnOperation() override {
    FunctionOp Function = getOperation();
    mlir::Region &Body = Function.getBody();

    if (Body.empty())
      return;

    mlir::GreedyRewriteConfig Config;
    Config.maxIterations = mlir::GreedyRewriteConfig::kNoLimit;
    if (mlir::applyPatternsAndFoldGreedily(Function, Patterns, Config).failed())
      signalPassFailure();
  }
};

} // namespace

PassPtr<FunctionOp> clift::createBooleanCastEliminationPass() {
  return std::make_unique<BooleanCastEliminationPass>();
}
