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

/// Turn the lvalue \p Value into an lvalue of type \p Type, the way C spells
/// it: take its address, retype the pointer and dereference it again.
mlir::Value castLvalue(mlir::PatternRewriter &Rewriter,
                       mlir::Location Loc,
                       mlir::Value Value,
                       mlir::Type Type,
                       uint64_t PointerSize) {
  auto From = PointerType::get(Value.getType(), PointerSize);
  auto To = PointerType::get(Type, PointerSize);

  mlir::Value Address = Rewriter.create<AddressofOp>(Loc, From, Value);
  Address = Rewriter.create<BitCastOp>(Loc, To, Address);
  return Rewriter.create<IndirectionOp>(Loc, Address);
}

/// Replace an access naming a segment field that holds a string with the
/// string itself. Both `a.b` and `a->b` can name one, so the pattern matches
/// the interface the two share.
struct EmitStringsPattern : mlir::OpInterfaceRewritePattern<AccessOpInterface> {
  RawBinaryView &BinaryView;
  uint64_t PointerSize;

  EmitStringsPattern(mlir::MLIRContext *Context,
                     RawBinaryView &BinaryView,
                     uint64_t PointerSize) :
    OpInterfaceRewritePattern(Context),
    BinaryView(BinaryView),
    PointerSize(PointerSize) {}

  mlir::LogicalResult
  matchAndRewrite(AccessOpInterface Access,
                  mlir::PatternRewriter &Rewriter) const override {
    mlir::Type Type = Access.getType();
    auto Array = mlir::dyn_cast<ArrayType>(Type);
    if (not Array or not isConst(Array))
      return mlir::failure();

    auto ElementType = mlir::dyn_cast<IntegerType>(Array.getElementType());
    if (not ElementType or ElementType.getKind() != IntegerKind::Unsigned)
      return mlir::failure();

    // Right now we only support number8_t strings
    unsigned CharacterSize = ElementType.getSize();
    if (CharacterSize != 1) {
      revng_log(Log, "Ignoring a string of " << CharacterSize << "-byte chars");
      return mlir::failure();
    }

    MetaAddress Address = getSegmentAddress(Access);
    if (not Address.isValid()) {
      revng_log(Log, "Ignoring an access landing at no known address");
      return mlir::failure();
    }

    uint64_t ElementsCount = Array.getElementsCount();
    uint64_t ByteCount = ElementsCount * CharacterSize;

    UnicodeCStringView String = readString(BinaryView,
                                           Address,
                                           ByteCount,
                                           CharacterSize);
    if (not String.isValid()) {
      revng_log(Log, "No string at " << Address.toString() << " after all");
      return mlir::failure();
    }

    revng_log(Log, "Emitting the string at " << Address.toString());

    // `readString` spans the terminator, which a `clift.str` leaves implicit.
    llvm::StringRef Characters = String.data().drop_back(CharacterSize);

    // A `clift.str` spells its characters as `number8_t`, while the field it
    // stands for is typed by the model and holds `uint8_t`.
    auto Character = IntegerType::get(getContext(),
                                      IntegerKind::Number,
                                      CharacterSize,
                                      /*IsConst=*/true);
    auto StringType = ArrayType::get(Character, ElementsCount);

    mlir::Location Loc = Access.getLoc();
    Rewriter.setInsertionPoint(Access);
    mlir::Value Literal = Rewriter.create<StringOp>(Loc,
                                                    StringType,
                                                    Characters);

    if (StringType == Type) {
      Rewriter.replaceOp(Access, Literal);
      return mlir::success();
    }

    // The cast goes on the pointer the only use decays the array to. An array
    // has to be reached through its address, so a use that is not a decay
    // falls back to casting the lvalue.
    auto Decay = mlir::dyn_cast<DecayOp>(getOnlyUse(Access)->getOwner());
    if (not Decay) {
      mlir::Value Cast = castLvalue(Rewriter, Loc, Literal, Type, PointerSize);
      Rewriter.replaceOp(Access, Cast);
      return mlir::success();
    }

    auto Pointer = PointerType::get(Character, PointerSize);
    mlir::Value Value = Rewriter.create<DecayOp>(Loc, Pointer, Literal);
    Rewriter.replaceOpWithNewOp<BitCastOp>(Decay, Decay.getType(), Value);
    Rewriter.eraseOp(Access);

    return mlir::success();
  }
};

} // namespace

namespace revng::pypeline::piperuns {

void EmitStrings::runOnCliftFunction(const model::Function &Function,
                                     clift::FunctionOp MLIRFunction) {
  mlir::MLIRContext *Context = MLIRFunction.getContext();

  mlir::RewritePatternSet Patterns(Context);
  Patterns.add<EmitStringsPattern>(Context,
                                   BinaryView,
                                   getDataModel(MLIRFunction).PointerSize);

  if (mlir::applyPatternsAndFoldGreedily(MLIRFunction, std::move(Patterns))
        .failed())
    revng_abort("emit-strings did not converge");
}

} // namespace revng::pypeline::piperuns
