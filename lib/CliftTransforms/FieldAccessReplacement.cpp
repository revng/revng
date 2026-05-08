//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/ADT/APInt.h"

#include "revng/Clift/Clift.h"
#include "revng/Clift/CliftTypes.h"

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
      Union,
      Struct,
      Array
    } TheKind;

    // We need to have the possibility to represent an `Index` with both a
    // constant and a variable component (in order to represent access like
    // `[i + 4]`)
    struct IndexInfo {
      mlir::Value Variable;
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
  bool replace(ExpressionOpInterface PointerToReplace,
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
      BaseType = mlir::cast<mlir::Type>(TypedefType.getUnderlyingType());
      continue;
    }

    // We should never reach these `Type`s by construction
    if (isa<FunctionType, PointerType, PrimitiveType, EnumType>(BaseType)) {
      revng_abort("Invalid type in traversal");
    }

    // Inspect `struct` or `union` (both implement ClassType)
    if (auto ClassType = mlir::dyn_cast<clift::ClassType>(BaseType)) {

      auto Kind = mlir::isa<clift::StructType>(BaseType) ?
                    FieldAccessInfo::Struct :
                    FieldAccessInfo::Union;
      unsigned FieldIndex = *FieldIt++;

      Result.FieldAccesses.push_back({ .TheKind = Kind,
                                       .Index = { mlir::Value(),
                                                  FieldIndex } });

      // Look up the field by positional index. Subtract the field's byte
      // offset from LeftoverOffset (for unions, getOffset() returns 0)
      const FieldAttr &Field = ClassType.getFields()[FieldIndex];
      BaseType = mlir::cast<mlir::Type>(Field.getType());
      LeftoverOffset.BaseOffset -= Field.getOffset();

      continue;
    }

    // Inspect the `array`
    if (auto ArrayType = mlir::dyn_cast<clift::ArrayType>(BaseType)) {

      ArrayShape CurrentArray = *ArrayIt++;

      // When reaching this iteration, if there was an array traversal in the
      // original traversal with a larger stride than the current, it must have
      // been already processed in a previous iteration
      if (not LeftoverOffset.LinearCombination.empty()) {
        revng_assert(LeftoverOffset.LinearCombination.front()
                       .Stride.ule(CurrentArray.Stride));
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

      mlir::Value DynamicElementId = {};
      const auto &LinearCombination = LeftoverOffset.LinearCombination;
      if (not LinearCombination.empty()
          and LinearCombination.front().Stride == CurrentArray.Stride) {
        auto &FrontTerm = LeftoverOffset.LinearCombination.front();
        DynamicElementId = FrontTerm.Idx.Variable;

        // If the Idx also has a constant component, add it to the fixed
        // consumed elements count
        if (FrontTerm.Idx.Constant.getBoolValue()) {
          NumFixedConsumedElements += FrontTerm.Idx.Constant;
        }

        LeftoverOffset.LinearCombination
          .erase(LeftoverOffset.LinearCombination.begin());
      }

      Result.FieldAccesses
        .push_back({ .TheKind = FieldAccessInfo::Array,
                     .Index = { DynamicElementId,
                                static_cast<uint64_t>(NumFixedConsumedElements
                                                        .getZExtValue()) } });

      // Move to the `array` element `Type`
      BaseType = ArrayType.getElementType();

      continue;
    }
  }

  // We pass over the remaining `LeftoverOffset`
  Result.LeftoverOffset = LeftoverOffset;

  return Result;
}

bool Replacement::replace(ExpressionOpInterface PointerToReplace,
                          const PointerArithmetic &Arithmetic) const {
  bool PropagatedThroughIndirection = false;

  // We need the `PointerSize` in order to generate the `ImmediateOp`s used to
  // access the `struct` fields and `array` members, and to generate the
  // `AddressOp` at the end of the field access substitution. We extract it
  // from the `PointerToReplace` we are processing.
  auto PointerSize = clift::unwrapped_cast<PointerType>(PointerToReplace
                                                          ->getResult(0)
                                                          .getType())
                       .getPointerSize();

  // Set insertion point right before the `PointerToReplace`
  mlir::OpBuilder Builder(PointerToReplace);
  mlir::Value CurrentValue = Arithmetic.BasePointer;

  // Every new `Operation` created in this phase will retain the `Location` of
  // the original `PointerToReplace`.
  // TODO: possible improvement for building the `PointerToReplaceLocation`.
  //       We could consider merging all the locations of all the
  //       `ExpressionOp`s that contributed to the computation of the
  //       `PointerArithmetic`. That would be much more accurate and probably
  //       give better results.
  mlir::Location PointerToReplaceLoc = PointerToReplace.getLoc();

  // Apply each field access in sequence
  // Iterate over every `FieldAccess` in `Replacement`, and materialize the
  // `clift` `Operation`s needed to perform such access
  for (const FieldAccessInfo &Access : FieldAccesses) {
    switch (Access.TheKind) {
    case FieldAccessInfo::Kind::Struct: {
      auto Index = Access.Index.Constant;
      auto [Type,
            IsIndirect] = getAccessedTypeInfo<clift::StructType>(CurrentValue);
      mlir::Type FieldType = Type.getFields()[Index].getType();
      CurrentValue = Builder.create<AccessOp>(PointerToReplaceLoc,
                                              FieldType,
                                              CurrentValue,
                                              IsIndirect,
                                              Index);
      break;
    }

    case FieldAccessInfo::Kind::Union: {
      auto Index = Access.Index.Constant;
      auto [Type,
            IsIndirect] = getAccessedTypeInfo<clift::UnionType>(CurrentValue);
      mlir::Type FieldType = Type.getFields()[Index].getType();
      CurrentValue = Builder.create<AccessOp>(PointerToReplaceLoc,
                                              FieldType,
                                              CurrentValue,
                                              IsIndirect,
                                              Index);
      break;
    }

    case FieldAccessInfo::Kind::Array: {

      // We may need to unwrap the `ArrayType` from a `PointerType`, and emit
      // the needed `IndirectionOp` and `Decay` cast accordingly.
      mlir::Type ArrayElementType;

      // We need to explicitly handle the `pointer as array` case, where
      // `CurrentValue` is not a `ptr<T>` of `ArrayType` (we virtually wrap it
      // ourselves), so the `indirection` and `cast<decay>` is not needed.
      if (auto P = unwrapped_dyn_cast<PointerType>(CurrentValue.getType());
          P and not unwrapped_isa<ArrayType>(P.getPointeeType())) {
        ArrayElementType = P.getPointeeType();
      } else {
        // Standard path emitting `indirection` and `cast<decay>` as needed
        auto [ArrayType,
              IsIndirect] = getAccessedTypeInfo<clift::ArrayType>(CurrentValue);
        if (IsIndirect) {
          CurrentValue = Builder.create<IndirectionOp>(PointerToReplaceLoc,
                                                       CurrentValue);
        }
        ArrayElementType = ArrayType.getElementType();
        auto DecayType = PointerType::get(ArrayElementType, PointerSize);
        CurrentValue = Builder.create<DecayOp>(PointerToReplaceLoc,
                                               DecayType,
                                               CurrentValue);
      }

      // Emit the `mlir::Value` representing the `Index` access.
      // We declare all the possible components (constant and variable parts)
      // as uninitialized here, and later fill only the components that we
      // need to emit.
      mlir::Value FixedIndexValue = {};
      mlir::Value DynamicIndexValue = {};
      mlir::Value IndexValue = {};

      // If present, we emit a new `mlir::Value` representing the constant
      // component of the `Index` access. If we do not have a `DynamicIndex`
      // component, we still emit a `imm 0` to represent the access to `[0]`.
      if (Access.Index.Constant != 0 or not Access.Index.Variable) {
        auto Index = Access.Index.Constant;
        auto IntegerType = Access.Index.Variable ?
                             Access.Index.Variable.getType() :
                             clift::IntegerType::get(Builder.getContext(),
                                                     IntegerKind::Generic,
                                                     PointerSize);
        FixedIndexValue = Builder.create<ImmediateOp>(PointerToReplaceLoc,
                                                      IntegerType,
                                                      Index);
      }

      // If present, we emit a new `mlir::Value` representing the variable
      // component of the `Index` access
      if (Access.Index.Variable) {
        DynamicIndexValue = Access.Index.Variable;
      }

      // We compose the constant and variable components of the `Index` access
      // depending on whether they are present
      if (FixedIndexValue and DynamicIndexValue) {
        IndexValue = Builder.create<AddOp>(PointerToReplaceLoc,
                                           FixedIndexValue,
                                           DynamicIndexValue);
      } else if (FixedIndexValue) {
        IndexValue = FixedIndexValue;
      } else {
        IndexValue = DynamicIndexValue;
      }

      // The `SubscriptOp` requires its `Pointer` operand to be a `PointerType`.
      // In case `CurrentValue` is a `PointerType`, wrapped into a
      // `TypeDefType`, we need to cast it to the underlying `PointerType`
      // first.
      if (not mlir::isa<clift::PointerType>(CurrentValue.getType())) {
        auto PtrType = unwrapped_cast<PointerType>(CurrentValue.getType());
        CurrentValue = Builder.create<BitCastOp>(PointerToReplaceLoc,
                                                 PtrType,
                                                 CurrentValue);
      }

      // Finally, we emit the `SubscriptOp` using as `Index` the `mlir::Value`
      // constructed above
      CurrentValue = Builder.create<SubscriptOp>(PointerToReplaceLoc,
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
  CurrentValue = Builder.create<AddressofOp>(PointerToReplaceLoc,
                                             CurrentValuePointerType,
                                             CurrentValue);

  // After we emit the `addressof`, we save the resulting `RichValue`, which may
  // contain a _rich_ type information of the emitted access. We collect this
  // here before the subsequent `cast` strips it of the type information`. We
  // then propagate the type information into `indirection` uses.
  mlir::Value RichValue = CurrentValue;

  // If there is a non-null `LeftoverOffset`, we add it as integer arithmetic
  if (not LeftoverOffset.BaseOffset.isZero()
      or not LeftoverOffset.LinearCombination.empty()) {

    // Cast pointer to integer
    auto IntPtrType = clift::IntegerType::get(Builder.getContext(),
                                              IntegerKind::Generic,
                                              PointerSize);
    CurrentValue = Builder.create<BitCastOp>(PointerToReplaceLoc,
                                             IntPtrType,
                                             CurrentValue);

    // Add base offset
    if (!LeftoverOffset.BaseOffset.isZero()) {

      auto LeftoverOffsetValue = LeftoverOffset.BaseOffset.getSExtValue();
      auto AddOperandValue = Builder.create<ImmediateOp>(PointerToReplaceLoc,
                                                         IntPtrType,
                                                         LeftoverOffsetValue);
      CurrentValue = Builder.create<AddOp>(PointerToReplaceLoc,
                                           CurrentValue,
                                           AddOperandValue);
    }

    // Add strided terms
    for (const auto &Term : LeftoverOffset.LinearCombination) {
      // Multiply stride by index
      auto IndexValue = Builder.create<ImmediateOp>(PointerToReplaceLoc,
                                                    IntPtrType,
                                                    Term.Idx.Constant
                                                      .getSExtValue());
      auto StrideValue = Builder
                           .create<ImmediateOp>(PointerToReplaceLoc,
                                                IntPtrType,
                                                Term.Stride.getSExtValue());
      auto StridedValue = Builder.create<clift::MulOp>(PointerToReplaceLoc,
                                                       IndexValue,
                                                       StrideValue);
      CurrentValue = Builder.create<clift::AddOp>(PointerToReplaceLoc,
                                                  CurrentValue,
                                                  StridedValue);
    }

    // Cast back to pointer
    auto PointerType = PointerType::get(IntPtrType, PointerSize);
    CurrentValue = Builder.create<BitCastOp>(PointerToReplaceLoc,
                                             PointerType,
                                             CurrentValue);
  }

  // If the result type differs from PointerToReplace's type, prepare a cast
  // to make the replacement fit the replaced `PointerToReplace` type.
  if (CurrentValue.getType() != PointerToReplace->getResult(0).getType()) {
    CurrentValue = Builder.create<BitCastOp>(PointerToReplaceLoc,
                                             PointerToReplace->getResult(0)
                                               .getType(),
                                             CurrentValue);
  }

  // Replace all the `Use`s of `PointerToReplace`, handling `IndirectionOp`s
  // specially: instead of giving them the "type-erased" `CurrentValue` (through
  // the `BitCast`) we give the `RichValue` directly so their result type
  // reflects the actual pointed-to type.
  if (CurrentValue != Arithmetic.BasePointer) {

    // Collect the `Use`s before the iteration, to avoid invalidation
    llvm::SmallVector<mlir::OpOperand *> Uses;
    for (auto &Use : PointerToReplace->getResult(0).getUses()) {
      Uses.push_back(&Use);
    }

    for (auto *Use : Uses) {

      // Check if we can propagate the _rich_ type through `indirection` uses.
      // We are interested in the `ptr<ptr<...>>` pattern, where the
      // `indirection` loads a pointer through which further field accesses can
      // be chained.
      // If we are inspecting a non-`IndirectionOp`, or if the propagation is
      // not applicable, we just redirect the `Use` to the type-matched
      // `CurrentValue`
      auto Indirection = mlir::dyn_cast<IndirectionOp>(Use->getOwner());
      auto RichPointerType = unwrapped_dyn_cast<PointerType>(RichValue
                                                               .getType());
      bool CanPropagate = RichPointerType
                          and unwrapped_isa<PointerType>(RichPointerType
                                                           .getPointeeType());
      if (not Indirection or not CanPropagate) {
        Use->set(CurrentValue);
        continue;
      }

      auto RichPointeeType = RichPointerType.getPointeeType();
      auto IndirectionResultType = Indirection.getResult().getType();

      // Size guard: the rich `pointee` must have the same byte size as the
      // `IndirectionOp` result for the retyping to be safe
      auto RichPointeeSize = getObjectSizeOrZero(RichPointeeType);
      auto IndirectionResultSize = getObjectSizeOrZero(IndirectionResultType);
      if (RichPointeeSize != IndirectionResultSize) {
        Use->set(CurrentValue);
        continue;
      }

      // Types already match — no propagation needed
      if (RichPointeeType == IndirectionResultType) {
        Use->set(CurrentValue);
        continue;
      }

      // Create a new typed `IndirectionOp` with the `RichValue` as operand,
      // so its result type is inferred from the `RichPointeeType`
      Builder.setInsertionPoint(Indirection);
      auto NewIndirection = Builder.create<IndirectionOp>(PointerToReplaceLoc,
                                                          RichValue);

      // Insert a `FixupCast` from the typed result to the old untyped result,
      // so existing users that expect the original type still verify
      auto FixupCast = Builder.create<BitCastOp>(PointerToReplaceLoc,
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

      if (not IndirectionUses.empty()) {
        PropagatedThroughIndirection = true;
      }

      // The old indirection's operand stays as-is (for assign LHS uses)
      Use->set(CurrentValue);
    }
  }

  // At this point, we are left in the `clift` IR with a set of dead `Value`s
  // representing the old `PointerArithmetic`. We rely on a subsequent DCE
  // pass to clean up all the dead `Value`s.

  // We signal at the above level that we propagated the `Type` info through an
  // `IndirectionOp`, which requires another iteration to find new rewriting
  // opportunities
  return PropagatedThroughIndirection;
}
} // namespace

/// Entry point function to perform the replacement of the pointer arithmetic
/// access (`PointerToReplace`), with operations equivalent to the
/// `BestTraversal` elected in the previous steps.
/// Returns `true` if a type was propagated through an `indirection`.
bool replaceFieldAccess(ExpressionOpInterface PointerToReplace,
                        const PointerArithmetic &Arithmetic,
                        const Traversal &BestTraversal) {

  // Derive the `PointerBitWidth` from the `PointerToReplace` type
  unsigned PointerBitWidth = clift::unwrapped_cast<PointerType>(PointerToReplace
                                                                  ->getResult(0)
                                                                  .getType())
                               .getPointerSize()
                             * 8;

  // We prepare the `Replacement`, which describes the `Traversal` in a way that
  // can easily be converted into a series of `clift` `operation`s
  auto R = Replacement::make(PointerBitWidth, Arithmetic, BestTraversal);

  // We actually perform the replacement
  return R.replace(PointerToReplace, Arithmetic);
}
