//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "revng/Clift/Clift.h"
#include "revng/CliftTransforms/EmitFieldAccesses.h"
#include "revng/CliftTransforms/Passes.h"

#include "EmitFieldAccesses/BestTraversal.h"
#include "EmitFieldAccesses/FieldAccessReplacement.h"
#include "EmitFieldAccesses/PointerArithmetic.h"

namespace clift {
#define GEN_PASS_DEF_CLIFTEMITFIELDACCESSES
#include "revng/CliftTransforms/Passes.h.inc"
} // namespace clift

using namespace clift;

namespace {

static thread_local TraversalInfoMap *ThreadTraversalCache;

struct EmitFieldAccessesPattern
  : mlir::OpInterfaceRewritePattern<ExpressionOpInterface> {

  EmitFieldAccessesPattern(mlir::MLIRContext *Context,
                           mlir::PatternBenefit Benefit = 1) :
    OpInterfaceRewritePattern(Context, Benefit) {}

  void initialize() { setDebugName("emit-field-accesses"); }

  mlir::LogicalResult
  matchAndRewrite(ExpressionOpInterface Op,
                  mlir::PatternRewriter &Rewriter) const override {
    std::optional<TraversalInfoMap> LocalTraversalCache;
    TraversalInfoMap *TraversalCache = ThreadTraversalCache;

    if (TraversalCache == nullptr)
      TraversalCache = &LocalTraversalCache.emplace();

    if (std::optional<PointerArithmetic> PA = computePointerArithmetic(Op)) {
      if (std::optional<Traversal> BT = computeBestTraversal(Op,
                                                             *PA,
                                                             *TraversalCache)) {

        if (replaceFieldAccess(Rewriter, Op, *PA, *BT))
          return mlir::success();
      }
    }

    return mlir::failure();
  }
};

struct EmitFieldAccessesPass
  : clift::impl::CliftEmitFieldAccessesBase<EmitFieldAccessesPass> {

  mlir::FrozenRewritePatternSet Patterns;

  mlir::LogicalResult initialize(mlir::MLIRContext *Context) override {
    mlir::RewritePatternSet Set(Context);
    populateWithEmitFieldAccessesPatterns(Set);
    Patterns = mlir::FrozenRewritePatternSet(std::move(Set));

    return mlir::success();
  }

  void runOnOperation() override {
    [[maybe_unused]] EFAThreadCache Cache;

    mlir::GreedyRewriteConfig Config;
    Config.maxIterations = mlir::GreedyRewriteConfig::kNoLimit;
    if (mlir::applyPatternsAndFoldGreedily(getOperation(), Patterns, Config)
          .failed())
      signalPassFailure();
  }
};

} // namespace

struct EFAThreadCache::ImplType {
  TraversalInfoMap Cache;

  ImplType() {
    revng_assert(ThreadTraversalCache == nullptr);
    ThreadTraversalCache = &Cache;
  }

  ~ImplType() {
    revng_assert(ThreadTraversalCache == &Cache);
    ThreadTraversalCache = nullptr;
  }
};

EFAThreadCache::EFAThreadCache() : Impl(std::make_unique<ImplType>()) {
}

EFAThreadCache::~EFAThreadCache() = default;

void populateWithEmitFieldAccessesPatterns(mlir::RewritePatternSet &Set) {
  Set.add<EmitFieldAccessesPattern>(Set.getContext());
}

PassPtr<FunctionOp> clift::createEmitFieldAccessesPass() {
  return std::make_unique<EmitFieldAccessesPass>();
}
