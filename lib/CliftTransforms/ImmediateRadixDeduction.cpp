//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/ADT/APSInt.h"
#include "llvm/ADT/DenseMap.h"

#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "revng/Clift/Clift.h"
#include "revng/CliftTransforms/Passes.h"

namespace clift {
#define GEN_PASS_DEF_CLIFTIMMEDIATERADIXDEDUCTION
#include "revng/CliftTransforms/Passes.h.inc"
} // namespace clift

using namespace clift;

namespace {

// This multiplier describes a weighing of the entropy comparison in favour of
// decimal (when the multiplier is greater than one).
static constexpr double HexEntropyMultiplier = 1.20;

template<uint64_t Radix>
static double computeRepresentationEntropy(uint64_t Value) {
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

// Deduces the best radix based purely the value of an integer immediate.
static unsigned deduceBestIntegerRadix(uint64_t Value) {
  if (Value < 0x10)
    return 10;

  if (Value > 64 and std::has_single_bit(Value))
    return 0x10;

  double DecEntropy = computeRepresentationEntropy<10>(Value);
  double HexEntropy = computeRepresentationEntropy<16>(Value);

  if (HexEntropy * HexEntropyMultiplier < DecEntropy)
    return 0x10;

  return 10;
}

static bool isBitwiseOperator(mlir::Operation *Op) {
  return mlir::isa<BitwiseNotOp, BitwiseAndOp, BitwiseOrOp, BitwiseXorOp>(Op);
}

// Deduces the best radix for an immediate based both on its usage and value.
static unsigned selectIntegerRadix(ImmediateOp Immediate) {
  for (mlir::Operation *User : Immediate.getResult().getUsers()) {
    // Operands of bitwise operators are always displayed in hexadecimal:
    if (isBitwiseOperator(User))
      return 0x10;
  }

  return deduceBestIntegerRadix(Immediate.getValue());
}

// For each integer immediate, deduce the best radix and if it is not 10 (the
// default), attach it in a discardable integer attribute named "clift.radix".
struct ImmediateRadixDeductionPattern : mlir::OpRewritePattern<ImmediateOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(ImmediateOp Immediate,
                  mlir::PatternRewriter &Rewriter) const override {
    if (Immediate->hasAttr("clift.radix"))
      return mlir::failure();

    unsigned Radix = selectIntegerRadix(Immediate);
    if (Radix == 10)
      return mlir::failure();

    auto RadixValue = llvm::APSInt(llvm::APInt(32, Radix));
    auto RadixAttr = mlir::IntegerAttr::get(getContext(), RadixValue);
    Immediate->setAttr("clift.radix", RadixAttr);

    return mlir::success();
  }
};

struct SwitchCaseRadixReductionPattern : mlir::OpRewritePattern<SwitchOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(SwitchOp Switch,
                  mlir::PatternRewriter &Rewriter) const override {
    if (Switch->hasAttr("clift.radix"))
      return mlir::failure();

    llvm::SmallDenseMap<unsigned, unsigned, 2> RadixCounts;
    for (uint64_t Value : Switch.getCaseValues())
      ++RadixCounts[deduceBestIntegerRadix(Value)];

    static constexpr unsigned Radices[] = { 10, 0x10 };

    unsigned BestRadix = 10;
    unsigned BestRadixCount = 0;

    for (unsigned Radix : Radices) {
      unsigned Count = RadixCounts[Radix];
      if (Count > BestRadixCount) {
        BestRadix = Radix;
        BestRadixCount = Count;
      }
    }

    // TODO: One walkAndApplyPatterns is available, it will not be necessary to
    //       add the attribute when the selected radix is 10.
    auto RadixValue = llvm::APSInt(llvm::APInt(32, BestRadix));
    auto RadixAttr = mlir::IntegerAttr::get(getContext(), RadixValue);
    Switch->setAttr("clift.radix", RadixAttr);

    return mlir::success();
  }
};

template<typename T>
using PassBase = clift::impl::CliftImmediateRadixDeductionBase<T>;

struct ImmediateRadixDeductionPass : PassBase<ImmediateRadixDeductionPass> {
  void runOnOperation() override {
    mlir::MLIRContext *Context = &getContext();

    mlir::RewritePatternSet Patterns(Context);
    Patterns.add<ImmediateRadixDeductionPattern>(Context);
    Patterns.add<SwitchCaseRadixReductionPattern>(Context);

    // TODO: Use walkAndApplyPatterns
    if (mlir::applyPatternsAndFoldGreedily(getOperation(), std::move(Patterns))
          .failed())
      signalPassFailure();
  }
};

} // namespace

PassPtr<FunctionOp> clift::createImmediateRadixDeductionPass() {
  return std::make_unique<ImmediateRadixDeductionPass>();
}
