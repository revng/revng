//
// This file is distributed under the MIT License. See LICENSE.md for details.
//
#include <compare>
#include <optional>

#include "revng/ADT/RecursiveCoroutine.h"
#include "revng/Clift/Clift.h"

#include "EmitFieldAccesses.h"
#include "PointerArithmetic.h"

namespace clift = mlir::clift;
using namespace clift;

static Logger Log("pointer-arithmetic");

// =============================================================================
// `PointerArithmetic` class methods
// =============================================================================

PointerArithmetic::OffsetExpression::OffsetExpression(unsigned BitWidth) :
  BaseOffset(BitWidth, 0) {
}

PointerArithmetic::OffsetExpression::OffsetExpression(llvm::APInt Offset) :
  BaseOffset(Offset) {
}

bool PointerArithmetic::isNumeric() const {
  return !BasePointer;
}

bool PointerArithmetic::isAddress() const {
  return static_cast<bool>(BasePointer);
}

bool PointerArithmetic::verify() const {

  const auto &LinearCombination = Offset.LinearCombination;

  // Check that strides are strictly positive. Negative or null strides do not
  // make sense
  for (const auto &Term : LinearCombination) {
    if (Term.Stride.isNegative() or Term.Stride.isZero()) {
      return false;
    }
  }

  // Check that we do not have duplicated strides, and that they are in
  // descending order
  for (size_t I = 0; I < LinearCombination.size(); I++) {
    for (size_t J = I + 1; J < LinearCombination.size(); J++) {
      if (LinearCombination[I].Stride == LinearCombination[J].Stride) {
        return false;
      }
      if (LinearCombination[I].Stride.ult(LinearCombination[J].Stride)) {
        return false;
      }
    }
  }

  // If we pass all the previous checks, we have a valid `PointerArithmetic`
  return true;
}

void PointerArithmetic::StridedTerm::dump() const {
  Log << "      Stride: ";
  Log << Stride.getZExtValue() << "\n";

  Log << "      Dynamic Index: ";
  if (Idx.Variable) {
    mlir::Value ValueToPrint = Idx.Variable;
    ValueToPrint.print(*Log.getAsLLVMStream());
  } else {
    Log << "(null)";
  }
  Log << "\n";

  Log << "      Fixed Offset: ";
  Log << Idx.Constant.getSExtValue() << "\n";
}

void PointerArithmetic::OffsetExpression::dump() const {
  Log << "  Base Offset: ";
  Log << BaseOffset.getSExtValue() << "\n";

  Log << "  Linear Combinations (" << LinearCombination.size() << " terms):\n";

  for (size_t I = 0; I < LinearCombination.size(); ++I) {
    Log << "    Term #" << I << ":\n";
    LinearCombination[I].dump();
  }
}

void PointerArithmetic::dump() const {

  Log << "Dumping PointerArithmetic object:\n";

  Log << "  BasePointer: ";
  if (BasePointer) {
    mlir::Value ValueToPrint = BasePointer;
    ValueToPrint.print(*Log.getAsLLVMStream());
    Log << "\n";
  } else {
    Log << "(null)\n";
  }

  Offset.dump();

  Log << "\n";
  Log.flush();
}

// =============================================================================
// `PointerArithmeticBuilder` class implementation, which contains the logic
// used to compute the resulting `PointerArithmetic`
// =============================================================================

/// The `PointerArithmeticBuilder` class is the main helper class used for
/// computing the `PointerArithmetic` object starting from the
/// `PointerToReplace`
class PointerArithmeticBuilder {
  unsigned PointerBitWidth = 0;

public:
  PointerArithmeticBuilder() = default;

  // Compute pointer arithmetic for a given `PointerToReplace`
  std::optional<PointerArithmetic>
  computePointerArithmetic(mlir::clift::ExpressionOpInterface PointerToReplace);

private:
  // Traversal function to inspect the operands. Uses RecursiveCoroutine
  // for stack safety, since traverse and compose* are mutually recursive
  // over user input expressions.
  RecursiveCoroutine<std::optional<PointerArithmetic>> traverse(mlir::Value V);

  // Return the `PointerArithmetic` for the leaf nodes
  PointerArithmetic createLeaf(mlir::Value V);

  // Methods which are used to compose the currently computed
  // `PointerArithmetic` with different `clift` `Operation`s that want to
  // traverse during our exploration
  RecursiveCoroutine<std::optional<PointerArithmetic>>
  composeBitcast(CastOp Bitcast);

  RecursiveCoroutine<std::optional<PointerArithmetic>> composeAdd(AddOp Add);

  RecursiveCoroutine<std::optional<PointerArithmetic>>
  composePtrAdd(PtrAddOp PtrAdd);

  RecursiveCoroutine<std::optional<PointerArithmetic>> composeMul(MulOp Mul);

  RecursiveCoroutine<std::optional<PointerArithmetic>>
  composeShl(ShiftLeftOp ShiftLeft);

  // Helper function used to multiply all the `PointerArithmetic` strides and
  // offset by a constant. Clears the BasePointer since the result of a
  // multiplication is no longer a pointer.
  PointerArithmetic multiplyByConstant(PointerArithmetic PA,
                                       const llvm::APInt &Multiplier);

  // Helper function used to sort linear combination by stride, in descending
  // order
  void sortLinearCombination(PointerArithmetic &PA);
};

// =============================================================================
// Static helper standalone functions
// =============================================================================

/// Helper function used to verify if an `ExpressionOpInterface` is
/// `PointerType`d
static bool isPointerTyped(ExpressionOpInterface Expr) {
  return Expr->getNumResults() > 0
         and isPointerType(Expr->getResult(0).getType());
}

/// Helper function used to extract the underlying constant value from an
/// `Value`
static std::optional<llvm::APInt> getConstantValue(mlir::Value V,
                                                   unsigned BitWidth) {
  if (auto Immediate = V.getDefiningOp<ImmediateOp>()) {
    return llvm::APInt(BitWidth, Immediate.getValue());
  }

  return std::nullopt;
}

// =============================================================================
// `PointerArithmeticBuilder` class methods
// =============================================================================

std::optional<PointerArithmetic>
PointerArithmeticBuilder::computePointerArithmetic(ExpressionOpInterface
                                                     PointerToReplace) {

  // We skip every non pointer-typed `PointerToReplace`
  if (not isPointerTyped(PointerToReplace)) {
    return std::nullopt;
  }

  // Derive the pointer bit size from the PointerToReplace type
  auto PtrType = getPointerType(PointerToReplace->getResult(0).getType());
  PointerBitWidth = PtrType.getPointerSize() * 8;

  // We traverse upwards the dataflow starting from the
  auto Result = rc_eval(traverse(PointerToReplace->getResult(0)));

  // If we got a numeric result, discard it, since we cannot use it
  if (Result and Result->isNumeric()) {
    return std::nullopt;
  }

  // Verify invariants for the obtained `PointerArithmetic`
  revng_assert(not Result or Result->verify());

  // Log the resulting `PointerArithmetic`, together with the initial
  // `PointerToReplace` it was produced from
  if (Log.isEnabled() and PointerToReplace and Result) {
    Log << "PointerArithmetic Relative to operation:\n";
    PointerToReplace.print(*Log.getAsLLVMStream());
    Log << "\n";
    Result->dump();
  }

  // If we reach this point, it means that we successfully computed the
  // `PointerArithmetic` object, so we can return it, so that the rewrite can
  // happen in later phases
  return Result;
}

RecursiveCoroutine<std::optional<PointerArithmetic>>
PointerArithmeticBuilder::traverse(mlir::Value V) {

  // Every time we find a compatible `Expression`, we traverse it in order to
  // compose its operands
  if (auto Expr = V.getDefiningOp<ExpressionOpInterface>()) {
    if (auto Cast = mlir::dyn_cast<CastOp>(Expr.getOperation())) {
      if (Cast.getKind() == CastKind::Bitcast) {
        rc_return rc_recur composeBitcast(Cast);
      }
    } else if (auto Add = mlir::dyn_cast<AddOp>(Expr.getOperation())) {
      rc_return rc_recur composeAdd(Add);
    } else if (auto PtrAdd = mlir::dyn_cast<PtrAddOp>(Expr.getOperation())) {
      rc_return rc_recur composePtrAdd(PtrAdd);
    } else if (auto Mul = mlir::dyn_cast<MulOp>(Expr.getOperation())) {
      rc_return rc_recur composeMul(Mul);
    } else if (auto Shl = mlir::dyn_cast<ShiftLeftOp>(Expr.getOperation())) {
      rc_return rc_recur composeShl(Shl);
    }
  }

  // If we do not traverse V, we create the leaf
  rc_return createLeaf(V);
}

PointerArithmetic PointerArithmeticBuilder::createLeaf(mlir::Value V) {
  using OffsetExpression = PointerArithmetic::OffsetExpression;
  PointerArithmetic PA{ .Offset = OffsetExpression(PointerBitWidth) };
  if (isPointerType(V.getType())) {

    // Pointer typed expression
    PA.BasePointer = V;
    PA.Offset = OffsetExpression(llvm::APInt(PointerBitWidth, 0));
  } else if (auto ConstantValue = getConstantValue(V, PointerBitWidth)) {

    // Integer Constant
    PA.Offset = OffsetExpression(*ConstantValue);
  } else {

    // We want to handle `ExpressionOpInterface`s and `mlir::BlockArgument`s for
    // building leaf `PA` containing a `LinearCombination`. These can be the
    // `Index` component of an array access.
    // Generic offset expression - strided 1 term.
    PA.Offset = OffsetExpression(llvm::APInt(PointerBitWidth, 0));
    PA.Offset.LinearCombination.emplace_back(llvm::APInt(PointerBitWidth, 1),
                                             PointerArithmetic::Index{
                                               V,
                                               llvm::APInt(PointerBitWidth,
                                                           0) });
  }

  return PA;
}

RecursiveCoroutine<std::optional<PointerArithmetic>>
PointerArithmeticBuilder::composeBitcast(CastOp Cast) {

  // Retrieve the `bitcast` single `Operand`, and recursively forward the
  // `PointerArithmetic` produced from it
  auto Operand = Cast->getOperand(0);

  auto OperandPA = rc_recur traverse(Operand);

  // Potentially, every `bitcast` could be treated as a `PointerOperand` leaf in
  // the exploration, if it was a `PointerType`. This can be useful in cases
  // where the exploration through the `Operand` produces a `Numeric`
  // `PointerArithmetic` (not reaching a `PointerType`). In such cases, we use
  // the current `bitcast` as a `BasePointer`. Better than nothing.
  if (OperandPA->isNumeric() and isPointerTyped(Cast)) {
    auto CastPA = createLeaf(Cast);
    rc_return CastPA;
  }

  rc_return OperandPA;
}

RecursiveCoroutine<std::optional<PointerArithmetic>>
PointerArithmeticBuilder::composeAdd(AddOp Add) {
  auto LHS = Add.getLhs();
  auto RHS = Add.getRhs();

  auto LHSPA = rc_recur traverse(LHS);
  auto RHSPA = rc_recur traverse(RHS);

  // It may happen that either the LHS or the RHS traversal produced an invalid
  // `PointerArithmetic` result. In such case, we need to propagate upward the
  // failure.
  if (not LHSPA or not RHSPA) {
    rc_return std::nullopt;
  }

  // It may happen that both `LHSPA` or `RHSPA` have an address
  // `PointerArithmetic`. In such situation, we cannot know which one is the
  // `BasePointer`, so we bail out from the construction of the
  // `PointerArithmetic` result.
  if (LHSPA->isAddress() and RHSPA->isAddress()) {
    rc_return std::nullopt;
  }

  // Add the two `PointerArithmetic`s together
  using OffsetExpression = PointerArithmetic::OffsetExpression;
  PointerArithmetic Result{ .Offset = OffsetExpression(PointerBitWidth) };

  // Add the `BaseOffsets`
  Result.Offset.BaseOffset = LHSPA->Offset.BaseOffset
                             + RHSPA->Offset.BaseOffset;

  // Append the linear combinations of `LHS` and `RHS`
  Result.Offset.LinearCombination = LHSPA->Offset.LinearCombination;
  Result.Offset.LinearCombination
    .append(RHSPA->Offset.LinearCombination.begin(),
            RHSPA->Offset.LinearCombination.end());

  // After we combined the `LinearCombination`s coming from the operands, we
  // sort them in descending order so that larger `Stride`s come before shorter
  // ones. This is the expected form derived from a nested array access, where
  // the stride of the _outer_ level is greater than the _inner_ one by
  // construction. Duplicate strides are checked in the verify phase as an
  // invariant.
  sortLinearCombination(Result);

  // After merging and sorting, if both sides contributed terms with the same
  // `Stride`, the `Result` would have duplicate `Stride`s and not pass verify.
  // In this situation, we return `nullopt` in order to soft fail the
  // `PointerArithmetic` construction, instead of not verifying.
  for (size_t I = 1; I < Result.Offset.LinearCombination.size(); ++I) {
    const auto &LinearCombination = Result.Offset.LinearCombination;
    if (LinearCombination[I].Stride == LinearCombination[I - 1].Stride) {
      rc_return std::nullopt;
    }
  }

  // We propagate as the `BasePointer` of the result the one corresponding to
  // the _address_ part of the traversal
  if (LHSPA->isAddress()) {
    Result.BasePointer = LHSPA->BasePointer;
  } else if (RHSPA->isAddress()) {
    Result.BasePointer = RHSPA->BasePointer;
  }

  rc_return Result;
}

RecursiveCoroutine<std::optional<PointerArithmetic>>
PointerArithmeticBuilder::composePtrAdd(PtrAddOp Add) {

  mlir::Value PointerOperand = Add.getPointer();
  mlir::Value OffsetOperand = Add.getOffset();

  // Compute the `PointerArithmetic` for both the pointer and offset operands.
  auto PointerOperandPA = rc_recur traverse(PointerOperand);
  auto OffsetOperandPA = rc_recur traverse(OffsetOperand);

  // We should check that the pointer and offset operands are not both `numeric`
  // or `address`
  if (not PointerOperandPA or not OffsetOperandPA) {
    rc_return std::nullopt;
  }
  revng_assert(not(PointerOperandPA->isAddress()
                   and OffsetOperandPA->isAddress()));
  revng_assert(not(PointerOperandPA->isNumeric()
                   and OffsetOperandPA->isNumeric()));

  // We expect that the offset operand is indeed a numeric `PointerArithmetic`.
  // We need this so that we can multiply the size of the clift pointee type by
  // the value that the numeric `PointerArithmetic` brings with it.
  revng_assert(OffsetOperandPA->isNumeric());

  // We multiply the size of the `PointeeType` of the `PointerOperand` operand
  // by the `OffsetOperand` `BaseOffset`. The multiplication factor is contained
  // into the `BaseOffset` of the numeric operand.
  auto PointerOperandType = getPointerType(PointerOperand.getType());
  auto PointeeSize = PointerOperandType.getPointeeType().getByteSize();
  PointerOperandPA->Offset.BaseOffset += OffsetOperandPA->Offset.BaseOffset
                                         * PointeeSize;

  rc_return PointerOperandPA;
}

RecursiveCoroutine<std::optional<PointerArithmetic>>
PointerArithmeticBuilder::composeMul(MulOp Mul) {
  auto LHS = Mul.getLhs();
  auto RHS = Mul.getRhs();

  // Categorize the `Mul` operands constant and non constant, so that we can
  // choose which one to traverse
  std::optional<llvm::APInt> Constant;
  mlir::Value Variable;

  if (auto C = getConstantValue(LHS, PointerBitWidth)) {
    Constant = C;
    Variable = RHS;
  } else if (auto C = getConstantValue(RHS, PointerBitWidth)) {
    Constant = C;
    Variable = LHS;
  } else {

    // Neither operand is constant, bail out
    rc_return std::nullopt;
  }

  // Traverse the variable operand
  auto VarPA = rc_recur traverse(Variable);
  if (not VarPA) {
    rc_return std::nullopt;
  }

  // Multiply the result (`multiplyByConstant` also clears `BasePointer`)
  rc_return multiplyByConstant(*VarPA, *Constant);
}

RecursiveCoroutine<std::optional<PointerArithmetic>>
PointerArithmeticBuilder::composeShl(ShiftLeftOp Shl) {

  // Handling is similar to `MulOp`, shift by N is multiplication by 2^N
  auto LHS = Shl.getLhs();
  auto RHS = Shl.getRhs();

  auto ShiftAmount = getConstantValue(RHS, PointerBitWidth);

  // We do not have a constant amount to perform the shift, so we bail out
  if (not ShiftAmount) {
    rc_return std::nullopt;
  }

  // We craft multiplication factor equivalent to the shift amount
  llvm::APInt Multiplier = llvm::APInt(PointerBitWidth, 1).shl(*ShiftAmount);

  // We traverse the variable operand
  auto LHSPA = rc_recur traverse(LHS);
  if (not LHSPA) {
    rc_return std::nullopt;
  }

  // Multiply the result (`multiplyByConstant` also clears `BasePointer`)
  rc_return multiplyByConstant(*LHSPA, Multiplier);
}

PointerArithmetic
PointerArithmeticBuilder::multiplyByConstant(PointerArithmetic PA,
                                             const llvm::APInt &Multiplier) {

  // Multiply the base offset
  PA.Offset.BaseOffset = PA.Offset.BaseOffset * Multiplier;

  // Multiply all the strides contained in the `LinearCombination`s
  for (auto &Term : PA.Offset.LinearCombination) {
    Term.Stride = Term.Stride * Multiplier;
  }

  // Multiplying implicitly returns something that is not a pointer
  PA.BasePointer = mlir::Value();

  return PA;
}

void PointerArithmeticBuilder::sortLinearCombination(PointerArithmetic &PA) {
  std::sort(PA.Offset.LinearCombination.begin(),
            PA.Offset.LinearCombination.end(),
            [](const auto &First, const auto &Second) {
              return First.Stride.ugt(Second.Stride);
            });
}

std::optional<PointerArithmetic>
computePointerArithmetic(mlir::clift::ExpressionOpInterface PointerToReplace) {
  auto Impl = PointerArithmeticBuilder();
  return Impl.computePointerArithmetic(PointerToReplace);
}
