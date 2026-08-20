//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/ADT/APInt.h"

#include "revng/Clift/Clift.h"
#include "revng/Clift/CliftTypes.h"
#include "revng/CliftTransforms/PatternRewriter.h"

#include "BestTraversal.h"
#include "FieldAccessReplacement.h"
#include "PointerArithmetic.h"

using namespace clift;

namespace {

/// Helper to unwrap a `clift` `Type` from a potential `PointerType` wrapper,
/// returning the unwrapped `Type` and whether indirection was needed
template<typename CliftType>
static std::pair<CliftType, bool>
getAccessedTypeInfo(mlir::Value CurrentValue) {
  if (auto P = clift::unwrapped_dyn_cast<PointerType>(CurrentValue.getType())) {
    return { clift::unwrapped_cast<CliftType>(P.getPointeeType()), true };
  }
  return { clift::unwrapped_cast<CliftType>(CurrentValue.getType()), false };
}

// =============================================================================
// `Replacement` struct definition
// =============================================================================

/// `Replacement` is used as a builder class to perform the rewrite of the
/// pointer arithmetic with (multiple) `clift` `operation`s equivalent to the
/// elected `BestTraversal`
struct Replacement {

  /// `FieldAccessInfo` represent the atomic element of the `Replacement`. It
  /// can represent an access to `union`, `struct`, `array`, accompanied by the
  /// relative `Index`
  struct FieldAccessInfo {
    enum Kind {
      Class,
      Array
    } TheKind;

    // An array index is a linear combination of runtime variables plus a
    // constant, `sum(Variables[k].Coefficient * Variables[k].Variable) +
    // Constant`, so that we can represent accesses like `[2*i + 3*j + 4]`. Each
    // coefficient maps a strided term's stride to the traversed array's element
    // stride, and is 1 for an exact stride match. For `struct`/`union` accesses
    // `Variables` is empty and `Constant` holds the field index.
    struct IndexInfo {
      struct ScaledVariable {
        uint64_t Coefficient;
        mlir::Value Variable;
      };
      llvm::SmallVector<ScaledVariable> Variables;
      uint64_t Constant;
    } Index;
  };

  /// We store the sequence of needed `FieldAccess`es here
  llvm::SmallVector<FieldAccessInfo> FieldAccesses;

  /// `LeftoverOffset` holds the eventual portion of the access that is not
  /// captured by the `BestTraversal`
  PointerArithmetic::OffsetExpression LeftoverOffset;

  /// Bit width of the pointer type being rewritten
  unsigned PointerBitWidth;

  /// This `static` method prepares the description of the `Replacement`, that
  /// will be later applied
  static Replacement make(unsigned PointerBitWidth,
                          const PointerArithmetic &Arithmetic,
                          const Traversal &BestTraversal);

  /// This method performs the actual `clift` IR rewriting. Returns `true` if
  /// a type was propagated through an indirection.
  bool replace(mlir::PatternRewriter &Rewriter,
               ExpressionOpInterface PointerToReplace,
               const PointerArithmetic &Arithmetic) const;
};

// =============================================================================
// `Replacement` methods implementation
// =============================================================================

/// Factory `make` constructor method
Replacement Replacement::make(unsigned PointerBitWidth,
                              const PointerArithmetic &Arithmetic,
                              const Traversal &BestTraversal) {

  auto BaseType = deriveBaseType(Arithmetic.BasePointer);

  // Start with an empty `Replacement` object, which will be populated in this
  // routine
  Replacement Result = {
    .LeftoverOffset = PointerArithmetic::OffsetExpression(PointerBitWidth),
    .PointerBitWidth = PointerBitWidth
  };

  // Copy the starting `BestTraversal` and `Offset`, we will consume them in the
  // current phase

  // We initialize the `LeftoverTraversal` with the `BestTraversal` we
  // identified during phase 2, and we consume it until the whole `Replacement`
  // is produced
  Traversal LeftoverTraversal = BestTraversal;

  // We initialize the `LeftoverOffset` with the offset expression that was
  // produced in the `PointerArithmetic` during phase 1
  PointerArithmetic::OffsetExpression LeftoverOffset = Arithmetic.Offset;

  // We perform an iterator-based traversal of the fields, going over each
  // component in the selected `Traversal` and building the `Replacement`
  auto FieldIt = LeftoverTraversal.TraversedFields.begin();
  auto FieldEnd = LeftoverTraversal.TraversedFields.end();
  auto ArrayIt = LeftoverTraversal.TraversedArrays.begin();
  auto ArrayEnd = LeftoverTraversal.TraversedArrays.end();
  while (FieldIt != FieldEnd or ArrayIt != ArrayEnd) {

    // Inspect the `TypedefType` and cast to a known `clift` `Type`
    if (auto TypedefType = mlir::dyn_cast<clift::TypedefType>(BaseType)) {
      BaseType = TypedefType.getUnderlyingType();
      continue;
    }

    // We should never reach these `Type`s by construction
    if (isa<FunctionType, PointerType, PrimitiveType, EnumType>(BaseType)) {
      revng_abort("Invalid type in traversal");
    }

    // Inspect `struct` or `union` (both implement ClassType)
    if (auto ClassType = mlir::dyn_cast<clift::ClassType>(BaseType)) {
      unsigned FieldIndex = *FieldIt++;
      Result.FieldAccesses.push_back({ .TheKind = FieldAccessInfo::Class,
                                       .Index = { {}, FieldIndex } });

      // Look up the field by positional index. Subtract the field's byte
      // offset from LeftoverOffset (for unions, getOffset() returns 0)
      const FieldAttr &Field = ClassType.getFields()[FieldIndex];
      BaseType = Field.getType();
      LeftoverOffset.BaseOffset -= Field.getOffset();

      continue;
    }

    // Inspect the `array`
    if (auto ArrayType = mlir::dyn_cast<clift::ArrayType>(BaseType)) {

      ArrayShape CurrentArray = *ArrayIt++;

      // This array consumes every remaining term whose stride it divides
      // (below); a term it does not divide is left for a later, smaller-stride
      // array. Selection (`canLowerArithmeticOntoTraversal`) guarantees some
      // array divides every term carrying a runtime index, so the front
      // (largest-stride) term, if it carries one, is either divisible by this
      // array or still has a later array to be visited; otherwise it would be
      // stranded here.
      if (not LeftoverOffset.LinearCombination.empty()
          and LeftoverOffset.LinearCombination.front().Idx.Variable) {
        const auto &Front = LeftoverOffset.LinearCombination.front();
        revng_assert(Front.Stride.urem(CurrentArray.Stride) == 0
                     or ArrayIt != ArrayEnd);
      }

      // We decide if we consume the offset from the `BaseOffset` or the
      // `LinearCombination`
      llvm::APInt NumFixedConsumedElements = llvm::APInt(PointerBitWidth, 0);
      if (LeftoverOffset.BaseOffset.uge(CurrentArray.Stride)) {
        NumFixedConsumedElements = LeftoverOffset.BaseOffset
                                     .udiv(CurrentArray.Stride);
        LeftoverOffset.BaseOffset -= CurrentArray.Stride
                                     * NumFixedConsumedElements;
      }

      // We consume every strided term this array's stride divides. A term
      // contributes `Stride * (Variable + Constant)` bytes, that is
      // `Coefficient * (Variable + Constant)` elements of this array, where
      // `Coefficient = Stride / CurrentArray.Stride`. Consuming all divisible
      // terms lets the array be indexed by a linear combination of the runtime
      // variables (e.g. `[2*i + j]`); an exact stride match is the coefficient
      // being 1.
      llvm::SmallVector<FieldAccessInfo::IndexInfo::ScaledVariable>
        IndexVariables;
      auto &LinearCombination = LeftoverOffset.LinearCombination;
      for (auto It = LinearCombination.begin();
           It != LinearCombination.end();) {
        if (It->Stride.urem(CurrentArray.Stride) != 0) {
          ++It;
          continue;
        }

        llvm::APInt Quotient = It->Stride.udiv(CurrentArray.Stride);
        if (It->Idx.Variable)
          IndexVariables.push_back({ Quotient.getZExtValue(),
                                     It->Idx.Variable });

        // The constant component is scaled by the same coefficient when
        // converting the term from stride units to array-element units.
        if (It->Idx.Constant.getBoolValue())
          NumFixedConsumedElements += It->Idx.Constant * Quotient;

        It = LinearCombination.erase(It);
      }

      Result.FieldAccesses
        .push_back({ .TheKind = FieldAccessInfo::Array,
                     .Index = { std::move(IndexVariables),
                                NumFixedConsumedElements.getZExtValue() } });

      // Move to the `array` element `Type`
      BaseType = ArrayType.getElementType();

      continue;
    }
  }

  // We pass over the remaining `LeftoverOffset`
  Result.LeftoverOffset = LeftoverOffset;

  return Result;
}

bool Replacement::replace(mlir::PatternRewriter &Rewriter,
                          ExpressionOpInterface PointerToReplace,
                          const PointerArithmetic &Arithmetic) const {
  auto IsZeroOffsetArrayAccess = [](const FieldAccessInfo &Access) {
    return Access.TheKind == FieldAccessInfo::Kind::Array
           and Access.Index.Variables.empty() and Access.Index.Constant == 0;
  };

  if (std::ranges::all_of(FieldAccesses, IsZeroOffsetArrayAccess))
    return false;

  mlir::Type PointerToReplaceType = PointerToReplace.getType();

  // We need the `PointerSize` in order to generate the `ImmediateOp`s used to
  // access the `struct` fields and `array` members, and to generate the
  // `AddressOp` at the end of the field access substitution. We extract it
  // from the `PointerToReplace` we are processing.
  auto PointerSize = clift::unwrapped_cast<PointerType>(PointerToReplaceType)
                       .getPointerSize();

  // Set insertion point right before the `PointerToReplace`
  mlir::Value CurrentValue = Arithmetic.BasePointer;

  // Every new `Operation` created in this phase will retain the `Location` of
  // the original `PointerToReplace`.
  // TODO: possible improvement for building the `PointerToReplaceLocation`.
  //       We could consider merging all the locations of all the
  //       `ExpressionOp`s that contributed to the computation of the
  //       `PointerArithmetic`. That would be much more accurate and probably
  //       give better results.
  mlir::Location PointerToReplaceLoc = PointerToReplace.getLoc();

  auto IntPtrType = clift::IntegerType::get(Rewriter.getContext(),
                                            IntegerKind::Generic,
                                            PointerSize);

  // Apply each field access in sequence
  // Iterate over every `FieldAccess` in `Replacement`, and materialize the
  // `clift` `Operation`s needed to perform such access
  for (const FieldAccessInfo &Access : FieldAccesses) {
    switch (Access.TheKind) {
    case FieldAccessInfo::Kind::Class: {
      auto Index = Access.Index.Constant;
      auto [Type, IsIndirect] = getAccessedTypeInfo<ClassType>(CurrentValue);
      mlir::Type FieldType = Type.getFields()[Index].getType();
      if (IsIndirect) {
        CurrentValue = Rewriter.create<IndirectAccessOp>(PointerToReplaceLoc,
                                                         FieldType,
                                                         CurrentValue,
                                                         Index);
      } else {
        CurrentValue = Rewriter.create<DirectAccessOp>(PointerToReplaceLoc,
                                                       FieldType,
                                                       CurrentValue,
                                                       Index);
      }
      break;
    }

    case FieldAccessInfo::Kind::Array: {
      // We need to explicitly handle the `pointer as array` case, where
      // `CurrentValue` is not a `ptr<T>` of `ArrayType` (we virtually wrap it
      // ourselves), so the `indirection` and `cast<decay>` is not needed.
      auto P = unwrapped_dyn_cast<PointerType>(CurrentValue.getType());
      if (not P or unwrapped_isa<ArrayType>(P.getPointeeType())) {
        // Standard path emitting `indirection` and `cast<decay>` as needed
        auto [ArrayType,
              IsIndirect] = getAccessedTypeInfo<clift::ArrayType>(CurrentValue);
        if (IsIndirect) {
          CurrentValue = Rewriter.create<IndirectionOp>(PointerToReplaceLoc,
                                                        CurrentValue);
        }
        auto DecayType = PointerType::get(ArrayType.getElementType(),
                                          PointerSize);
        CurrentValue = Rewriter.create<DecayOp>(PointerToReplaceLoc,
                                                DecayType,
                                                CurrentValue);
      }

      // Emit the array index as the linear combination
      // `Constant + sum(Coefficient * Variable)`. We emit the constant first
      // (or a bare `imm` when the index has no variable part, e.g. a `[0]` or
      // `[n]` access), then add each scaled variable component.
      mlir::Value IndexValue = {};

      auto AddTerm =
        [&Rewriter, &PointerToReplaceLoc, &IndexValue](mlir::Value Term) {
          if (IndexValue) {
            IndexValue = Rewriter.create<AddOp>(PointerToReplaceLoc,
                                                IndexValue,
                                                Term);
          } else {
            IndexValue = Term;
          }
        };

      // If a constant offset is present, an immediate operation is emitted
      // to represent it.
      if (Access.Index.Constant != 0 or Access.Index.Variables.empty()) {
        AddTerm(Rewriter.create<ImmediateOp>(PointerToReplaceLoc,
                                             IntPtrType,
                                             Access.Index.Constant));
      }

      for (const auto &Term : Access.Index.Variables) {
        mlir::Value Contribution = Term.Variable;

        if (not equivalent(Contribution.getType(), IntPtrType)) {
          Contribution = Rewriter.create<BitCastOp>(PointerToReplaceLoc,
                                                    IntPtrType,
                                                    Contribution);
        }

        if (Term.Coefficient != 1) {
          auto Coefficient = Rewriter.create<ImmediateOp>(PointerToReplaceLoc,
                                                          IntPtrType,
                                                          Term.Coefficient);

          Contribution = Rewriter.create<MulOp>(PointerToReplaceLoc,
                                                Contribution,
                                                Coefficient);
        }

        AddTerm(Contribution);
      }

      // Finally, we emit a `SubscriptOp` to represent the arithmetic on
      // `CurrentValue` and the `IndexValue` created above.
      CurrentValue = Rewriter.create<SubscriptOp>(PointerToReplaceLoc,
                                                  CurrentValue,
                                                  IndexValue);

      break;
    }
    }
  }

  // Take address of the result, since we always start the replacement from a
  // `PointerType`, we want to get back to it
  auto CurrentValuePointerType = PointerType::get(CurrentValue.getType(),
                                                  PointerSize);
  CurrentValue = Rewriter.create<AddressofOp>(PointerToReplaceLoc,
                                              CurrentValuePointerType,
                                              CurrentValue);

  // After we emit the `addressof`, we save the resulting `RichType`, which may
  // contain a _rich_ type information of the emitted access. We collect this
  // here before the subsequent `cast` strips it of the type information`. We
  // then propagate the type information into `indirection` uses.
  mlir::Type RichType = CurrentValue.getType();

  // If there is a non-null `LeftoverOffset`, we add it as integer arithmetic
  if (not LeftoverOffset.BaseOffset.isZero()
      or not LeftoverOffset.LinearCombination.empty()) {

    // Cast pointer to integer
    CurrentValue = Rewriter.create<BitCastOp>(PointerToReplaceLoc,
                                              IntPtrType,
                                              CurrentValue);

    // Add base offset
    if (!LeftoverOffset.BaseOffset.isZero()) {

      auto Operand = Rewriter.create<ImmediateOp>(PointerToReplaceLoc,
                                                  IntPtrType,
                                                  LeftoverOffset.BaseOffset);
      CurrentValue = Rewriter.create<AddOp>(PointerToReplaceLoc,
                                            CurrentValue,
                                            Operand);
    }

    // Add strided terms
    for (const auto &Term : LeftoverOffset.LinearCombination) {
      // Traversal selection (`canLowerArithmeticOntoTraversal`) guarantees that
      // every runtime index was consumed into an array subscript, so any term
      // left over here is a pure constant. A variable index would be silently
      // dropped by the emission below, so we enforce the invariant explicitly.
      revng_assert(not Term.Idx.Variable);

      // Multiply stride by index
      auto IndexValue = Rewriter.create<ImmediateOp>(PointerToReplaceLoc,
                                                     IntPtrType,
                                                     Term.Idx.Constant);
      auto StrideValue = Rewriter.create<ImmediateOp>(PointerToReplaceLoc,
                                                      IntPtrType,
                                                      Term.Stride);
      auto StridedValue = Rewriter.create<clift::MulOp>(PointerToReplaceLoc,
                                                        IndexValue,
                                                        StrideValue);
      CurrentValue = Rewriter.create<clift::AddOp>(PointerToReplaceLoc,
                                                   CurrentValue,
                                                   StridedValue);
    }

    // Cast back to pointer
    CurrentValue = Rewriter.create<BitCastOp>(PointerToReplaceLoc,
                                              RichType,
                                              CurrentValue);
  }

  auto AsType = [&Rewriter,
                 &PointerToReplaceLoc](mlir::Value Value,
                                       mlir::Type Type) -> mlir::Value {
    if (Value.getType() != Type) {
      Value = Rewriter.create<BitCastOp>(PointerToReplaceLoc, Type, Value);
    }
    return Value;
  };

  // Replace all the `Use`s of `PointerToReplace`, handling `IndirectionOp`s
  // specially: instead of giving them the "type-erased" `CurrentValue` (through
  // the `BitCast`) we give the `RichValue` directly so their result type
  // reflects the actual pointed-to type.
  if (CurrentValue != Arithmetic.BasePointer) {

    // Collect the `Use`s before the iteration, to avoid invalidation
    llvm::SmallVector<mlir::OpOperand *> Uses;
    for (auto &Use : PointerToReplace.getResult().getUses()) {
      Uses.push_back(&Use);
    }

    for (mlir::OpOperand *Use : Uses) {

      // Check if we can propagate the _rich_ type through `indirection` uses.
      // We are interested in the `ptr<ptr<...>>` pattern, where the
      // `indirection` loads a pointer through which further field accesses can
      // be chained.
      // If we are inspecting a non-`IndirectionOp`, or if the propagation is
      // not applicable, we just redirect the `Use` to the type-matched
      // `CurrentValue`
      auto Indirection = mlir::dyn_cast<IndirectionOp>(Use->getOwner());
      auto RichPointerType = unwrapped_dyn_cast<PointerType>(RichType);
      bool CanPropagate = RichPointerType
                          and unwrapped_isa<PointerType>(RichPointerType
                                                           .getPointeeType());
      if (not Indirection or not CanPropagate) {
        setOperandValue(Rewriter,
                        *Use,
                        AsType(CurrentValue, PointerToReplaceType));
        continue;
      }

      auto RichPointeeType = RichPointerType.getPointeeType();
      auto IndirectionResultType = Indirection.getType();

      // Size guard: the rich `pointee` must have the same byte size as the
      // `IndirectionOp` result for the retyping to be safe
      auto RichPointeeSize = getObjectSizeOrZero(RichPointeeType);
      auto IndirectionResultSize = getObjectSizeOrZero(IndirectionResultType);
      if (RichPointeeSize != IndirectionResultSize) {
        setOperandValue(Rewriter,
                        *Use,
                        AsType(CurrentValue, PointerToReplaceType));
        continue;
      }

      // Types already match — no propagation needed
      if (RichPointeeType == IndirectionResultType) {
        setOperandValue(Rewriter,
                        *Use,
                        AsType(CurrentValue, PointerToReplaceType));
        continue;
      }

      // Create a new typed `IndirectionOp` with the `RichValue` as operand,
      // so its result type is inferred from the `RichPointeeType`
      Rewriter.setInsertionPoint(Indirection);
      auto NewIndirection = Rewriter.create<IndirectionOp>(PointerToReplaceLoc,
                                                           AsType(CurrentValue,
                                                                  RichType));

      // Insert a `FixupCast` from the typed result to the old untyped result,
      // so existing users that expect the original type still verify
      auto FixupCast = Rewriter.create<BitCastOp>(PointerToReplaceLoc,
                                                  IndirectionResultType,
                                                  NewIndirection);

      // Redirect the old `IndirectionOp`'s uses to the `FixupCast`.
      // Lvalue uses (assign LHS) stay on the old `IndirectionOp`: a `BitCastOp`
      // is not an lvalue, and changing the LHS type would break the assign's
      // type constraints. The old `IndirectionOp` is left for DCE.
      llvm::SmallVector<mlir::OpOperand *> IndirectionUses;
      for (auto &IndirUse : Indirection.getResult().getUses()) {
        if (mlir::isa<AssignOp>(IndirUse.getOwner())
            and IndirUse.getOperandNumber() == 0) {
          continue;
        }
        IndirectionUses.push_back(&IndirUse);
      }

      for (auto *IndirectionUse : IndirectionUses) {
        IndirectionUse->set(FixupCast);
      }

      // The old indirection's operand stays as-is (for assign LHS uses)
      setOperandValue(Rewriter,
                      *Use,
                      AsType(CurrentValue, PointerToReplaceType));
    }
  }

  // At this point, we are left in the `clift` IR with a set of dead `Value`s
  // representing the old `PointerArithmetic`. We rely on a subsequent DCE
  // pass to clean up all the dead `Value`s.

  return true;
}
} // namespace

/// Entry point function to perform the replacement of the pointer arithmetic
/// access (`PointerToReplace`), with operations equivalent to the
/// `BestTraversal` elected in the previous steps.
/// Returns `true` if the IR was modified.
bool replaceFieldAccess(mlir::PatternRewriter &Rewriter,
                        ExpressionOpInterface PointerToReplace,
                        const PointerArithmetic &Arithmetic,
                        const Traversal &BestTraversal) {

  // Derive the `PointerBitWidth` from the `PointerToReplace` type
  unsigned PointerBitWidth = clift::unwrapped_cast<PointerType>(PointerToReplace
                                                                  .getType())
                               .getPointerSize()
                             * 8;

  // We prepare the `Replacement`, which describes the `Traversal` in a way that
  // can easily be converted into a series of `clift` `operation`s
  auto R = Replacement::make(PointerBitWidth, Arithmetic, BestTraversal);

  // We actually perform the replacement
  return R.replace(Rewriter, PointerToReplace, Arithmetic);
}
