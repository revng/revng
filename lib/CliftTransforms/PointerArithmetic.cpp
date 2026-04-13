//
// This file is distributed under the MIT License. See LICENSE.md for details.
//
#include <compare>
#include <optional>

#include "revng/ADT/RecursiveCoroutine.h"
#include "revng/Clift/Clift.h"

#include "EmitFieldAccesses.h"
#include "PointerArithmetic.h"

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
  computePointerArithmetic(clift::ExpressionOpInterface PointerToReplace);

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
  composeBitcast(BitCastOp Bitcast);

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

  // Helper to combine two `PointerArithmetic`s that are being added together.
  // Checks validity (both non-null, not both address), creates the Result with
  // combined BaseOffset and propagated BasePointer. Returns nullopt on failure.
  std::optional<PointerArithmetic>
  combineAddOperands(const std::optional<PointerArithmetic> &LHSPA,
                     const std::optional<PointerArithmetic> &RHSPA);

  // Merge `LinearCombination`s from LHS and RHS into `Result`, sort by
  // descending `Stride`, and check for duplicates. Returns `false` if
  // duplicated `Stride`s are found.
  using StridedTermVector = llvm::SmallVector<PointerArithmetic::StridedTerm>;
  bool mergeLinearCombinations(PointerArithmetic &Result,
                               const StridedTermVector &LHSTerms,
                               const StridedTermVector &RHSTerms);

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
  if (mlir::isa<YieldOp>(Expr.getOperation()))
    return false;

  return clift::unwrapped_isa<clift::PointerType>(Expr->getResult(0).getType());
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
  auto PtrType = clift::unwrapped_cast<PointerType>(PointerToReplace
                                                      ->getResult(0)
                                                      .getType());
  PointerBitWidth = PtrType.getPointerSize() * 8;

  // We traverse upwards the dataflow starting from the
  auto Result = rc_eval(traverse(PointerToReplace->getResult(0)));

  // If we got a numeric result, discard it, since we cannot use it
  if (Result and Result->isNumeric()) {
    return std::nullopt;
  }

  // We skip every replacement where the `PointerToReplace` is equal to the
  // `BasePointer`. Replacing such expression would create a circular reference:
  // the new `clift.subscript` uses the `BasePointer` as operand, and
  // `replaceAllUsesWith` would replace the `BasePointer` uses with the new
  // `clift.subscript` result, and this insert the circular reference between
  // mlir `Value`s.
  if (PointerToReplace->getResult(0) == Result->BasePointer) {

    // We expect this situation only for empty `Result->Offset`
    const auto &Offset = Result->Offset;
    revng_assert(Offset.BaseOffset.isZero()
                 and Offset.LinearCombination.empty());
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
    if (auto Cast = mlir::dyn_cast<BitCastOp>(Expr.getOperation())) {
      rc_return rc_recur composeBitcast(Cast);
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
  if (clift::unwrapped_isa<PointerType>(V.getType())) {

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
PointerArithmeticBuilder::composeBitcast(BitCastOp Cast) {

  // Retrieve the `bitcast` single `Operand`, and recursively forward the
  // `PointerArithmetic` produced from it
  auto Operand = Cast->getOperand(0);

  rc_return rc_recur traverse(Operand);
}

/// Helper to combine two `PointerArithmetic`s that are being added together.
/// Checks validity (both non-null, not both address), creates the `Result` with
/// combined `BaseOffset` and propagate the `BasePointer` of the `Address`
/// component. Returns `nullopt` on failure.
using PAB = PointerArithmeticBuilder;
std::optional<PointerArithmetic>
PAB::combineAddOperands(const std::optional<PointerArithmetic> &LHSPA,
                        const std::optional<PointerArithmetic> &RHSPA) {

  // Either operand traversal failed
  if (not LHSPA or not RHSPA) {
    return std::nullopt;
  }

  // Both operands are address — ambiguous BasePointer
  if (LHSPA->isAddress() and RHSPA->isAddress()) {
    return std::nullopt;
  }

  // Build the combined result
  using OffsetExpression = PointerArithmetic::OffsetExpression;
  PointerArithmetic Result{ .Offset = OffsetExpression(PointerBitWidth) };

  // Add the `BaseOffsets`
  Result.Offset.BaseOffset = LHSPA->Offset.BaseOffset
                             + RHSPA->Offset.BaseOffset;

  // Propagate the `BasePointer` from whichever operand is an address
  if (LHSPA->isAddress()) {
    Result.BasePointer = LHSPA->BasePointer;
  } else if (RHSPA->isAddress()) {
    Result.BasePointer = RHSPA->BasePointer;
  }

  return Result;
}

/// Merge `LinearCombinations` from `LHS` and `RHS` into Result and sort the by
/// descending `Stride`. If duplicates are found, we return `false`.
bool PAB::mergeLinearCombinations(PointerArithmetic &Result,
                                  const StridedTermVector &LHSTerms,
                                  const StridedTermVector &RHSTerms) {

  Result.Offset.LinearCombination = LHSTerms;
  Result.Offset.LinearCombination.append(RHSTerms.begin(), RHSTerms.end());

  // Sort in descending order so that larger `Stride`s come before shorter
  // ones. This is the expected form derived from a nested array access, where
  // the stride of the _outer_ level is greater than the _inner_ one by
  // construction.
  sortLinearCombination(Result);

  // After merging and sorting, if both sides contributed terms with the same
  // `Stride`, the `Result` would have duplicate `Stride`s and not pass verify.
  for (size_t I = 1; I < Result.Offset.LinearCombination.size(); ++I) {
    const auto &LC = Result.Offset.LinearCombination;
    if (LC[I].Stride == LC[I - 1].Stride) {
      return false;
    }
  }

  return true;
}

RecursiveCoroutine<std::optional<PointerArithmetic>>
PointerArithmeticBuilder::composeAdd(AddOp Add) {
  auto LHSPA = rc_recur traverse(Add.getLhs());
  auto RHSPA = rc_recur traverse(Add.getRhs());

  auto Result = combineAddOperands(LHSPA, RHSPA);
  if (not Result) {
    rc_return std::nullopt;
  }

  // The `mergeLinearCombinations` helper will return `false` when duplicates
  // are found, and we should bail out in that situation
  if (not mergeLinearCombinations(*Result,
                                  LHSPA->Offset.LinearCombination,
                                  RHSPA->Offset.LinearCombination)) {
    rc_return std::nullopt;
  }

  rc_return *Result;
}

RecursiveCoroutine<std::optional<PointerArithmetic>>
PointerArithmeticBuilder::composePtrAdd(PtrAddOp PtrAdd) {

  auto PointerOperandPA = rc_recur traverse(PtrAdd.getPointer());
  auto OffsetOperandPA = rc_recur traverse(PtrAdd.getOffset());

  auto Result = combineAddOperands(PointerOperandPA, OffsetOperandPA);
  if (not Result) {
    rc_return std::nullopt;
  }

  // For `ptr_add`, the offset is in units of the pointee type, so we scale
  // the `OffsetOperand` contributions by the `PointeeType` size
  auto PointerType = clift::unwrapped_cast<clift::PointerType>(PtrAdd
                                                                 .getPointer()
                                                                 .getType());
  auto PointeeSize = getObjectSize(PointerType.getPointeeType());

  // Scale `BaseOffset` of the `OffsetOperandPA`
  Result->Offset.BaseOffset = PointerOperandPA->Offset.BaseOffset
                              + OffsetOperandPA->Offset.BaseOffset
                                  * PointeeSize;

  // Scale the `OffsetOperand` `LinearCombination` strides by `PointeeSize`
  llvm::SmallVector<PointerArithmetic::StridedTerm> ScaledOffsetTerms;
  for (const auto &Term : OffsetOperandPA->Offset.LinearCombination) {
    ScaledOffsetTerms.emplace_back(Term.Stride * PointeeSize, Term.Idx);
  }

  // The `mergeLinearCombinations` helper will return `false` when duplicates
  // are found, and we should bail out in that situation
  if (not mergeLinearCombinations(*Result,
                                  PointerOperandPA->Offset.LinearCombination,
                                  ScaledOffsetTerms)) {
    rc_return std::nullopt;
  }

  rc_return *Result;
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
computePointerArithmetic(clift::ExpressionOpInterface PointerToReplace) {
  auto Impl = PointerArithmeticBuilder();
  return Impl.computePointerArithmetic(PointerToReplace);
}
