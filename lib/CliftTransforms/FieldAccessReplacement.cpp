//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/ADT/APInt.h"

#include "revng/Clift/Clift.h"
#include "revng/Clift/CliftTypes.h"

#include "BestTraversal.h"
#include "FieldAccessReplacement.h"
#include "PointerArithmetic.h"

namespace clift = mlir::clift;
using namespace mlir::clift;

namespace {

/// Helper to unwrap a `clift` `Type` from a potential `PointerType` wrapper,
/// returning the unwrapped `Type` and whether indirection was needed
template<typename CliftType>
static std::pair<CliftType, bool>
getAccessedTypeInfo(mlir::Value CurrentValue) {
  if (isPointerType(CurrentValue.getType())) {
    auto PtrType = getPointerType(CurrentValue.getType());
    return { mlir::cast<CliftType>(dealias(PtrType.getPointeeType())), true };
  }
  return { mlir::cast<CliftType>(dealias(CurrentValue.getType())), false };
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

  /// This method performs the actual `clift` IR rewriting,
  void replace(ExpressionOpInterface PointerToReplace,
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

void Replacement::replace(ExpressionOpInterface PointerToReplace,
                          const PointerArithmetic &Arithmetic) const {

  // We need the `PointerSize` in order to generate the `ImmediateOp`s used to
  // access the `struct` fields and `array` members, and to generate the
  // `AddressOp` at the end of the field access substitution. We extract it
  // from the `PointerToReplace` we are processing.
  auto PointerSize = getPointerType(PointerToReplace->getResult(0).getType())
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
      clift::ValueType ArrayElementType;

      // We need to explicitly handle the `pointer as array` case, where
      // `CurrentValue` is not a `ptr<T>` of `ArrayType` (we virtually wrap it
      // ourselves), so the `indirection` and `cast<decay>` is not needed.
      if (auto PtrType = getPointerType(CurrentValue.getType());
          PtrType
          and not mlir::isa<ArrayType>(dealias(PtrType.getPointeeType(),
                                               /*IgnoreQualifiers=*/true))) {
        ArrayElementType = PtrType.getPointeeType();
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
        CurrentValue = Builder.create<CastOp>(PointerToReplaceLoc,
                                              DecayType,
                                              CurrentValue,
                                              CastKind::Decay);
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
        auto IntegerType = PrimitiveType::get(Builder.getContext(),
                                              PrimitiveKind::GenericKind,
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

  // If there is a non-null `LeftoverOffset`, we add it as integer arithmetic
  if (not LeftoverOffset.BaseOffset.isZero()
      or not LeftoverOffset.LinearCombination.empty()) {

    // Cast pointer to integer
    auto IntegerType = PrimitiveType::get(Builder.getContext(),
                                          PrimitiveKind::GenericKind,
                                          PointerSize);
    CurrentValue = Builder.create<CastOp>(PointerToReplaceLoc,
                                          IntegerType,
                                          CurrentValue,
                                          CastKind::Bitcast);

    // Add base offset
    if (!LeftoverOffset.BaseOffset.isZero()) {
      auto IntegerType = PrimitiveType::get(Builder.getContext(),
                                            PrimitiveKind::GenericKind,
                                            PointerSize);

      auto LeftoverOffsetValue = LeftoverOffset.BaseOffset.getSExtValue();
      auto AddOperandValue = Builder.create<ImmediateOp>(PointerToReplaceLoc,
                                                         IntegerType,
                                                         LeftoverOffsetValue);
      CurrentValue = Builder.create<AddOp>(PointerToReplaceLoc,
                                           CurrentValue,
                                           AddOperandValue);
    }

    // Add strided terms
    for (const auto &Term : LeftoverOffset.LinearCombination) {
      // Multiply stride by index
      auto IndexValue = Builder.create<ImmediateOp>(PointerToReplaceLoc,
                                                    IntegerType,
                                                    Term.Idx.Constant
                                                      .getSExtValue());
      auto StrideValue = Builder
                           .create<ImmediateOp>(PointerToReplaceLoc,
                                                IntegerType,
                                                Term.Stride.getSExtValue());
      auto StridedValue = Builder.create<clift::MulOp>(PointerToReplaceLoc,
                                                       IndexValue,
                                                       StrideValue);
      CurrentValue = Builder.create<clift::AddOp>(PointerToReplaceLoc,
                                                  CurrentValue,
                                                  StridedValue);
    }

    // Cast back to pointer
    auto PointerType = PointerType::get(CurrentValue.getType(), PointerSize);
    CurrentValue = Builder.create<CastOp>(PointerToReplaceLoc,
                                          PointerType,
                                          CurrentValue,
                                          CastKind::Bitcast);
  }

  // If the result type differs from PointerToReplace's type, add a cast on
  // `clift` IR in order to make the replacement fit the replaced
  // `PointerToReplace` `Type`
  auto PointerToReplaceOp = PointerToReplace.getOperation();
  if (CurrentValue.getType() != PointerToReplace->getResult(0).getType()) {
    CurrentValue = Builder
                     .create<CastOp>(PointerToReplaceLoc,
                                     PointerToReplace->getResult(0).getType(),
                                     CurrentValue,
                                     CastKind::Bitcast);
  }

  // Replace all uses of `PointerToReplace` with `CurrentValue`
  if (CurrentValue != Arithmetic.BasePointer) {
    PointerToReplace->getResult(0).replaceAllUsesWith(CurrentValue);
  }

  // At this point, we are left in the `clift` IR with a set of dead `Value`s
  // representing the old `PointerArithmetic`. We rely on a subsequent DCE
  // pass to clean up all the dead `Value`s.
}
} // namespace

/// Entry point function to perform the replacement of the pointer arithmetic
/// access (`PointerToReplace`), with operations equivalent to the
/// `BestTraversal` elected in the previous steps
void replaceFieldAccess(ExpressionOpInterface PointerToReplace,
                        const PointerArithmetic &Arithmetic,
                        const Traversal &BestTraversal) {

  // Derive the `PointerBitWidth` from the `PointerToReplace` type
  unsigned PointerBitWidth = getPointerType(PointerToReplace->getResult(0)
                                              .getType())
                               .getPointerSize()
                             * 8;

  // We prepare the `Replacement`, which describes the `Traversal` in a way that
  // can easily be converted into a series of `clift` `operation`s
  auto R = Replacement::make(PointerBitWidth, Arithmetic, BestTraversal);

  // We actually perform the replacement
  R.replace(PointerToReplace, Arithmetic);
}
