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

/// \return the size of a character of the string \p Type spells or 0 if it
/// doesn't.
unsigned getCharacterSize(mlir::Type Type) {
  auto Array = mlir::dyn_cast<ArrayType>(Type);
  if (not Array or not isConst(Array))
    return 0;

  auto Character = mlir::dyn_cast<IntegerType>(Array.getElementType());
  if (not Character or Character.getKind() != IntegerKind::Unsigned)
    return 0;

  if (Character.getSize() != 1 and Character.getSize() != 2)
    return 0;

  return Character.getSize();
}

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

  void initialize() { setDebugName("emit-strings"); }

  mlir::LogicalResult
  matchAndRewrite(AccessOpInterface Access,
                  mlir::PatternRewriter &Rewriter) const override {
    mlir::Operation *Operation = Access.getOperation();
    mlir::Value Result = Operation->getResult(0);
    mlir::Type Type = Result.getType();

    unsigned CharacterSize = getCharacterSize(Type);
    if (CharacterSize == 0)
      return mlir::failure();

    // Right now we only support number8_t strings
    if (CharacterSize != 1) {
      revng_log(Log, "Ignoring a string of " << CharacterSize << "-byte chars");
      return mlir::failure();
    }

    MetaAddress Address = getSegmentAddress(Result);
    if (not Address.isValid()) {
      revng_log(Log, "Ignoring an access landing at no known address");
      return mlir::failure();
    }

    uint64_t ElementsCount = mlir::cast<ArrayType>(Type).getElementsCount();
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
    Rewriter.setInsertionPoint(Operation);
    mlir::Value Literal = Rewriter.create<StringOp>(Loc,
                                                    StringType,
                                                    Characters);

    if (StringType == Type) {
      Rewriter.replaceOp(Operation, Literal);
      return mlir::success();
    }

    // The cast goes on the pointer the only use decays the array to. An array
    // has to be reached through its address, so a use that is not a decay
    // falls back to casting the lvalue.
    auto Decay = mlir::dyn_cast<DecayOp>(getOnlyUse(Result)->getOwner());
    if (not Decay) {
      mlir::Value Cast = castLvalue(Rewriter, Loc, Literal, Type, PointerSize);
      Rewriter.replaceOp(Operation, Cast);
      return mlir::success();
    }

    auto Pointer = PointerType::get(Character, PointerSize);
    mlir::Value Value = Rewriter.create<DecayOp>(Loc, Pointer, Literal);
    Rewriter.replaceOpWithNewOp<BitCastOp>(Decay, Decay.getType(), Value);
    Rewriter.eraseOp(Operation);

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
