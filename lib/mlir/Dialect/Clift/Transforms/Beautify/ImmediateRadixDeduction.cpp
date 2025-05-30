//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/ADT/APSInt.h"

#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "revng/mlir/Dialect/Clift/IR/CliftOps.h"
#include "revng/mlir/Dialect/Clift/Transforms/Passes.h"

namespace mlir {
namespace clift {
#define GEN_PASS_DEF_CLIFTIMMEDIATERADIXDEDUCTION
#include "revng/mlir/Dialect/Clift/Transforms/Passes.h.inc"
} // namespace clift
} // namespace mlir

namespace clift = mlir::clift;

namespace {

template<uint64_t Radix>
static double uniqueDigitsToRadixRatio(uint64_t Value) {
  revng_assert(Value != 0);

  uint8_t Digits[Radix] = {};
  uint8_t UniqueDigitCount = 0;
  uint8_t DigitCount = 0;

  do {
    uint8_t Digit = Value % Radix;
    Value = Value / Radix;
    UniqueDigitCount += Digits[Digit]++ == 0;
    ++DigitCount;
  } while (Value != 0);

  return static_cast<double>(UniqueDigitCount) / DigitCount;
}

static unsigned deduceIntegerRadix(uint64_t Value) {
  if (Value < 0x10)
    return 10;

  if (Value > 64 and std::has_single_bit(Value))
    return 0x10;

  double DecRatio = uniqueDigitsToRadixRatio<10>(Value);
  double HexRatio = uniqueDigitsToRadixRatio<16>(Value);

  if (DecRatio / HexRatio >= 1.20)
    return 0x10;

  return 10;
}

struct ImmediateRadixDeductionPattern :
  mlir::OpRewritePattern<clift::ImmediateOp> {

  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(clift::ImmediateOp Immediate,
                  mlir::PatternRewriter &Rewriter) const override {
    if (Immediate->hasAttr("clift.radix"))
      return mlir::failure();

    auto Radix = deduceIntegerRadix(Immediate.getValue());
    if (Radix == 10)
      return mlir::failure();

    auto RadixValue = llvm::APSInt(llvm::APInt(32, Radix));
    auto RadixAttr = mlir::IntegerAttr::get(getContext(), RadixValue);
    Immediate->setAttr("clift.radix", RadixAttr);

    return mlir::success();
  }
};

template<typename T>
using PassBase = mlir::clift::impl::CliftImmediateRadixDeductionBase<T>;

struct ImmediateRadixDeductionPass : PassBase<ImmediateRadixDeductionPass> {
  void runOnOperation() override {
    mlir::MLIRContext *Context = &getContext();

    mlir::RewritePatternSet Patterns(Context);
    Patterns.add<ImmediateRadixDeductionPattern>(Context);

    // TODO: Use walkAndApplyPatterns
    if (mlir::applyPatternsAndFoldGreedily(getOperation(), std::move(Patterns))
          .failed())
      signalPassFailure();
  }
};

} // namespace

clift::PassPtr<clift::FunctionOp> clift::createImmediateRadixDeductionPass() {
  return std::make_unique<ImmediateRadixDeductionPass>();
}

