//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "revng/Clift/Clift.h"
#include "revng/Clift/CliftOpHelpers.h"
#include "revng/Clift/LocationAddresses.h"
#include "revng/CliftPipes/EmitStrings.h"
#include "revng/Model/Binary.h"
#include "revng/SegmentReferences/StringConstants.h"
#include "revng/Support/Debug.h"

using namespace clift;

static Logger Log("emit-strings");

namespace {

static MetaAddress getGlobalObjectMetaAddress(mlir::Value Value) {
  uint64_t Offset = 0;

  while (true) {
    mlir::Operation *Op = Value.getDefiningOp();
    if (Op == nullptr)
      return MetaAddress::invalid();

    if (auto E = mlir::dyn_cast<UseOp>(Op)) {
      auto Global = mlir::cast<GlobalVariableOp>(E.getUsedGlobal());

      auto Location = pipeline::locationFromString(revng::ranks::Segment,
                                                   Global.getHandle());
      if (not Location)
        return MetaAddress::invalid();

      return std::get<0>(Location->at(revng::ranks::Segment)) + Offset;
    }

    if (auto E = mlir::dyn_cast<DirectAccessOp>(Op)) {
      Value = E.getValue();
      Offset += E.getFieldAttr().getOffset();
      continue;
    }

    return MetaAddress::invalid();
  }
}

struct EmitStringsPattern : mlir::OpRewritePattern<DirectAccessOp> {
  RawBinaryView &BinaryView;

  EmitStringsPattern(mlir::MLIRContext *Context, RawBinaryView &BinaryView) :
    OpRewritePattern(Context), BinaryView(BinaryView) {}

  mlir::LogicalResult
  matchAndRewrite(DirectAccessOp Access,
                  mlir::PatternRewriter &Rewriter) const override {
    auto ArrayType = mlir::dyn_cast<clift::ArrayType>(Access.getType());
    if (not ArrayType)
      return mlir::failure();

    auto CodeUnitType = mlir::dyn_cast<IntegerType>(ArrayType.getElementType());
    if (not CodeUnitType or not isConst(CodeUnitType)
        or CodeUnitType.getKind() != IntegerKind::Unsigned)
      return mlir::failure();

    uint64_t CodeUnitSize = CodeUnitType.getSize();

    // For now, only 8-bit code units are supported.
    if (CodeUnitSize != 1)
      return mlir::failure();

    MetaAddress Address = getGlobalObjectMetaAddress(Access);
    if (not Address.isValid()) {
      revng_log(Log, "Ignoring an access landing at no known address");
      return mlir::failure();
    }

    UnicodeCStringView String = //
      readString(BinaryView,
                 Address,
                 CodeUnitSize * ArrayType.getElementsCount(),
                 CodeUnitSize);

    if (not String.isValid()) {
      revng_log(Log, "No string at " << Address.toString() << " after all");
      return mlir::failure();
    }

    revng_log(Log, "Emitting the string at " << Address.toString());

    // Drop the null terminator, which is not part of the string content.
    llvm::StringRef Content = String.data().drop_back(CodeUnitSize);

    Rewriter.setInsertionPoint(Access);
    Rewriter.replaceOpWithNewOp<StringOp>(Access, ArrayType, Content);

    return mlir::success();
  }
};

} // namespace

namespace revng::pypeline::piperuns {

void EmitStrings::runOnCliftFunction(const model::Function &Function,
                                     clift::FunctionOp MLIRFunction) {
  mlir::MLIRContext *Context = MLIRFunction.getContext();

  mlir::RewritePatternSet Patterns(Context);
  Patterns.add<EmitStringsPattern>(Context, BinaryView);

  // TODO: Use walkAndApplyPatterns
  if (mlir::applyPatternsAndFoldGreedily(MLIRFunction, std::move(Patterns))
        .failed())
    revng_abort("emit-strings did not converge");
}

} // namespace revng::pypeline::piperuns
