//
// This file is distributed under the MIT License. See LICENSE.md for details.
//
#include <compare>
#include <cstdint>
#include <limits>
#include <optional>

#include "revng/ADT/RecursiveCoroutine.h"
#include "revng/Clift/Clift.h"
#include "revng/Support/Assert.h"

#include "BestTraversal.h"
#include "PointerArithmetic.h"

namespace clift = mlir::clift;
using namespace clift;

static Logger Log("best-traversal");

/// Helper function which converts a generic `ArrayPath` to a compatible form
/// used to store the `array` traversal into the `Traversal` class. The
/// re-ordering in descending `Stride` order is provided by the comparison
/// operator of `ArrayShape`
static llvm::SmallVector<ArrayShape>
arrayPathToSortedVector(const ArrayPath &Path) {
  llvm::SmallVector<ArrayShape> Result;
  Result.reserve(Path.size());
  for (const NestedArrayShape &Nested : Path) {
    ArrayShape Shape;
    Shape.NumElements = Nested.NumElements;
    Shape.Stride = Nested.Stride;
    Result.push_back(Shape);
  }
  llvm::sort(Result);
  return Result;
}

// =============================================================================
// `ArrayShape` class methods
// =============================================================================

// =============================================================================
// `Traversal` class methods
// =============================================================================

Traversal::Traversal(mlir::Type TargetType,
                     int64_t StartOffset,
                     int64_t LeftoverOffset,
                     std::vector<uint64_t> TraversedFields,
                     const ArrayPath &Path) :
  TargetType(TargetType),
  StartOffset(StartOffset),
  LeftoverOffset(LeftoverOffset),
  TraversedFields(std::move(TraversedFields)),
  TraversedArrays(arrayPathToSortedVector(Path)) {
}

int64_t Traversal::depth() const {
  return TraversedArrays.size() + TraversedFields.size();
}

int64_t Traversal::begin() const {
  return StartOffset + LeftoverOffset;
}

int64_t Traversal::end() const {
  return begin() + getObjectSizeOrZero(TargetType);
}

void Traversal::dump() const {

  Log << "\nDumping Traversal:\n";

  // We dump the `TargetType` on which the `Traversal` lands on
  Log << "  TargetType: ";
  TargetType.print(*Log.getAsLLVMStream());
  Log << "\n";

  // Dump the `StartOffset` and `LeftoverOffset`
  Log << "  StartOffset: " << StartOffset << "\n";
  Log << "  LeftoverOffset: " << LeftoverOffset << "\n";

  // Dump all the `Traversed Fields` which this `Traversal` describes
  Log << "  Traversed Fields (" << TraversedFields.size() << "): [";
  for (size_t I = 0; I < TraversedFields.size(); ++I) {
    if (I > 0)
      Log << ", ";
    Log << TraversedFields[I];
  }
  Log << "]\n";

  // Dump all the `TraversedArray` which this `Traversal` describes
  Log << "  Traversed Arrays (" << TraversedArrays.size() << "):\n";
  for (const auto &Array : TraversedArrays) {
    Log << "    { NumElements: " << Array.NumElements
        << ", Stride: " << Array.Stride << " }\n";
  }

  Log << "\n";
  Log.flush();
}

mlir::Type deriveBaseType(mlir::Value BasePointer) {

  auto BasePtrType = unwrapped_cast<PointerType>(BasePointer.getType());
  auto PointeeType = BasePtrType.getPointeeType();
  auto UnderlyingType = unwrapTypedefs(PointeeType);

  // If the pointee is a struct, union, or array, use it directly — the
  // traversal analyzer can walk its fields and arrays.
  // Wrapping also a `struct`, would mean rewriting a constant offset access
  // into it as `p[0].field` instead of `p->field`.
  if (mlir::isa<StructType, UnionType, ArrayType>(UnderlyingType)) {
    return PointeeType;
  }

  // We don't want to wrap `void` or `function` `Type`s into `array`s.
  if (not unwrapped_isa<ObjectType>(PointeeType)) {
    return PointeeType;
  }

  // In all the other situations, wrap into an implicit array. We use
  // `ImplicitArrayNumElements` a _very large_ array in order to cover any
  // reasonable constant offset.
  static constexpr uint64_t
    ImplicitArrayNumElements = std::numeric_limits<uint64_t>::max();
  return ArrayType::get(PointeeType, ImplicitArrayNumElements);
}

namespace {

// =============================================================================
// Static helper functions
// =============================================================================

/// Helper function which checks if the `BaseOffset` falls inside of the
/// innermost `array` described by the `ArrayPath`
static bool isCompatible(const ArrayPath &Path, llvm::APInt BaseOffset) {
  for (const NestedArrayShape &Shape : Path) {

    // If the offset doesn't reach the offset of this `Shape` inside its parent
    // we just bail out
    if (BaseOffset.ult(Shape.OffsetFromParentArrayElement)) {
      return false;
    }

    BaseOffset -= Shape.OffsetFromParentArrayElement;

    // If we're entering in an array element past the first, we have to adjust
    // `BaseOffset`, consuming it
    if (BaseOffset.uge(Shape.Stride)) {

      // If we're jumping over the whole array, past it, we just bail out.
      if (BaseOffset.udiv(Shape.Stride).uge(Shape.NumElements)) {
        return false;
      }

      // Otherwise adjust the `BaseOffset`, to set it to the offset inside the
      // array element we're traversing
      BaseOffset = BaseOffset.urem(Shape.Stride);
    }
  }

  // If we reach this point, it means that the `ArrayPath` was inedeed
  // compatible
  return true;
}

/// Helper function which finds all the `ArrayPath`s that are compatible with a
/// given `BaseOffset`
static llvm::SmallVector<const ArrayPath *>
findCompatibleArrayPaths(const std::vector<ArrayPath> &AllArrayPaths,
                         const llvm::APInt &BaseOffset) {
  llvm::SmallVector<const ArrayPath *> CompatibleArrays;
  for (const ArrayPath &Path : AllArrayPaths) {
    if (isCompatible(Path, BaseOffset))
      CompatibleArrays.push_back(&Path);
  }
  return CompatibleArrays;
}

/// Helper function which counts the length of the common prefix between two
/// sorted `TraversedArrays` vectors. Since both are sorted in descending stride
/// order, this counts how many leading array shapes match exactly.
static uint64_t commonPrefixStrides(const llvm::ArrayRef<ArrayShape> &LHS,
                                    const llvm::ArrayRef<ArrayShape> &RHS) {
  uint64_t Count = 0;
  auto LIt = LHS.begin();
  auto RIt = RHS.begin();

  while (LIt != LHS.end() && RIt != RHS.end() && *LIt == *RIt) {
    ++Count;
    ++LIt;
    ++RIt;
  }

  return Count;
}

/// The `typeDistance` function computes the distance between two `Type`s.
/// Returns 0 if the types are equal (after stripping `typedef`s), or infinity
/// otherwise. The return type is kept as `uint64_t` to allow future refinement
/// of the scoring criterion, enabling more fine grained control over this
/// score.
static uint64_t typeDistance(mlir::Type Explicit, mlir::Type Ideal) {

  // Unwrap any typedefs to compare the underlying types
  Explicit = unwrapTypedefs(Explicit);
  Ideal = unwrapTypedefs(Ideal);

  return Explicit == Ideal ? 0 : std::numeric_limits<uint64_t>::max();
}

// =============================================================================
// `Score` class definition
// =============================================================================

/// We represent the `SizeRelation` between two `Traversal`s
enum class SizeRelation {
  Same,
  Larger,
  Smaller,
  DontCare
};

/// `Score` is used to represent the score obtained comparing two `Traversal`s.
/// It embeds the criteria we define in order to select the `BestTraversal` that
/// we want to use to rewrite the pointer access.
struct Score {
  bool Valid;
  int64_t StartDistance;
  SizeRelation SizeRelation;
  uint64_t TypeDistance;
  uint64_t CommonStrides;
  int64_t Depth;

  static Score invalid();

  std::strong_ordering operator<=>(const Score &Other);
};

Score Score::invalid() {
  return Score{ .Valid = false,
                .StartDistance = 0,
                .SizeRelation = SizeRelation::DontCare,
                .TypeDistance = 0,
                .CommonStrides = 0,
                .Depth = 0 };
}

/// We redefine the spaceship operator in order to define the ordering criteria
/// for comparing `Score`s, which drives the selection of the `BestTraversal`
std::strong_ordering Score::operator<=>(const Score &Other) {

  // An `Invalid` field must be considered `greater` than a `Valid` one
  if (not Valid and Other.Valid) {
    return std::strong_ordering::greater;
  }
  if (Valid and not Other.Valid) {
    return std::strong_ordering::less;
  }
  if (not Valid and not Other.Valid) {
    return std::strong_ordering::equal;
  }

  // When both `Score`s are valid, we move on to comparing:
  // 1) `StartDistance`
  auto Cmp = StartDistance <=> Other.StartDistance;
  if (Cmp != 0) {
    return Cmp;
  }

  // 2) `SizeRelation`
  Cmp = SizeRelation <=> Other.SizeRelation;
  if (Cmp != 0) {
    return Cmp;
  }

  // 3) `TypeDistance`
  Cmp = TypeDistance <=> Other.TypeDistance;
  if (Cmp != 0) {
    return Cmp;
  }

  // 4) `CommonStrides`
  Cmp = Other.CommonStrides <=> CommonStrides;
  if (Cmp != 0) {
    return Cmp;
  }

  // 5) `Depth`
  return Depth <=> Other.Depth;
}

/// The `score` function is used in order to obtain a _similarity_ `Score`
/// between the `Explicit` and `Ideal` `Traversal`s. We want to select the
/// `Traversal` with the minimal score as the one that will constitute the
/// pointer access rewrite
static Score score(const Traversal &Explicit, const Traversal &Ideal) {
  int64_t StartDistance = Explicit.begin() - Ideal.begin();
  int64_t EndDistance = Explicit.end() - Ideal.end();

  uint64_t CommonStrides = commonPrefixStrides(Explicit.TraversedArrays,
                                               Ideal.TraversedArrays);
  uint64_t TypeDistValue = typeDistance(Explicit.TargetType, Ideal.TargetType);

  if (StartDistance < 0) {

    // Explicit comes first - invalid
    return Score::invalid();
  } else if (StartDistance == 0) {

    // They start at same point
    if (EndDistance == 0) {

      // We have a perfect match, in this situation we will rely on the other
      // criteria to elect the `BestTraversal`
      return Score{ .Valid = true,
                    .StartDistance = 0,
                    .SizeRelation = SizeRelation::Same,
                    .TypeDistance = TypeDistValue,
                    .CommonStrides = CommonStrides,
                    .Depth = Ideal.depth() };
    } else if (EndDistance < 0) {

      // Explicit ends before Ideal
      return Score{ .Valid = true,
                    .StartDistance = 0,
                    .SizeRelation = SizeRelation::Larger,
                    .TypeDistance = 0,
                    .CommonStrides = 0,
                    .Depth = Ideal.depth() };
    } else if (EndDistance > 0) {

      // Explicit ends after Ideal
      return Score{ .Valid = true,
                    .StartDistance = 0,
                    .SizeRelation = SizeRelation::Smaller,
                    .TypeDistance = 0,
                    .CommonStrides = 0,
                    .Depth = Ideal.depth() };
    }
  } else if (StartDistance > 0) {

    // Ideal comes first (StartDistance > 0)
    if (EndDistance <= 0) {

      // Explicit ends before or at Ideal
      return Score{ .Valid = true,
                    .StartDistance = StartDistance,
                    .SizeRelation = SizeRelation::DontCare,
                    .TypeDistance = 0,
                    .CommonStrides = 0,
                    .Depth = Ideal.depth() };
    } else {

      // Explicit ends after Ideal - partial overlap, invalid
      return Score::invalid();
    }
  }

  // We do not expect that we can reach this point, all the previous case should
  // cover all the possibilities
  revng_abort();
}

// =============================================================================
// `TypeTraversalAnalyzer` class definition
// =============================================================================

/// `TypeTraversalAnalyzer` is used as an oracle to compute and retrieve
/// `Traversal`s and `ArrayPath`s in a lazy manner
class TypeTraversalAnalyzer {
private:
  /// Reference to the cache containing the pre-computed `Traversal`s and
  /// `ArrayPath`s. This passed in the constructor, as we want to cache the
  /// information across different runs
  TraversalInfoMap &Data;

public:
  TypeTraversalAnalyzer(TraversalInfoMap &Data) : Data(Data) {}

public:
  /// Retrieve the precomputed `Traversal`s for `BaseType` (or compute them
  /// on-the-fly`)
  const std::vector<Traversal> &getTraversals(mlir::Type BaseType);

  /// Retrieve the precomputed `ArrayPath`s for `BaseType` (or compute them
  /// on-the-fly`)
  const std::vector<ArrayPath> &getArrayPaths(mlir::Type BaseType);

  /// Retrieve only the slice of `Traversal`s, starting from the `BaseType`,
  /// that are useful for the comparison with the current access described by
  /// `Arithmetic`.
  /// There are two _modes of operation_:
  /// 1) When `SmartLookup` is off, we simply retrieve all the `Traversal`s
  ///    relative to `BaseType`. With this mode of operation, the additional
  ///    parameters are actually useless.
  //// 2) When `SmartLookup` is on, we employ some smarties in order to reduce
  ///     the search space for `Traversal`s that we retrieve.
  //      TODO: this mode of operation is not implemented ATM.
  std::pair<std::vector<Traversal>::const_iterator,
            std::vector<Traversal>::const_iterator>
  getTraversalRange(mlir::Type BaseType,
                    const PointerArithmetic &Arithmetic,
                    mlir::Type PointeeType,
                    bool SmartLookup);

private:
  /// Entry point for traversing the `BaseType`, and producing the corresponding
  /// `Traversal`s and `ArrayPaths` (takes care of ordering them)
  llvm::DenseMap<mlir::Type, TraversalInfo>::iterator
  traverse(mlir::Type BaseType);

  /// Underlying `impl` method for performing the recursive step of the traverse
  /// of a `BaseType`. Uses `RecursiveCoroutine` for stack safety.
  RecursiveCoroutine<void>
  traverseImpl(mlir::Type Type,
               std::vector<Traversal> &Traversals,
               std::vector<ArrayPath> &ArrayPaths,
               int64_t CurrentOffset = 0,
               const std::vector<uint64_t> &FieldPath = {},
               const ArrayPath &CurrentArrayPath = {});
};

const std::vector<Traversal> &
TypeTraversalAnalyzer::getTraversals(mlir::Type BaseType) {
  auto It = Data.find(BaseType);
  if (It == Data.end()) {
    It = traverse(BaseType);
  }

  return It->second.Traversals;
}

const std::vector<ArrayPath> &
TypeTraversalAnalyzer::getArrayPaths(mlir::Type BaseType) {
  auto It = Data.find(BaseType);
  if (It == Data.end()) {
    It = traverse(BaseType);
  }

  return It->second.ArrayPaths;
}

std::pair<std::vector<Traversal>::const_iterator,
          std::vector<Traversal>::const_iterator>
TypeTraversalAnalyzer::getTraversalRange(mlir::Type BaseType,
                                         const PointerArithmetic &Arithmetic,
                                         mlir::Type PointeeType,
                                         bool SmartLookup) {

  // TODO: for the time being, we do not implement the smart lookup logic, and
  //       therefore we assert that this does not happen until we have added the
  //       implementation
  if (not SmartLookup) {
    const std::vector<Traversal> &Traversals = getTraversals(BaseType);
    return { Traversals.begin(), Traversals.end() };
  }

  revng_abort("Fast lookup not implemented");
}

llvm::DenseMap<mlir::Type, TraversalInfo>::iterator
TypeTraversalAnalyzer::traverse(mlir::Type BaseType) {
  revng_assert(Data.count(BaseType) == 0);

  auto [It, Inserted] = Data.insert({ BaseType, TraversalInfo() });
  auto &[Traversals, ArrayPaths] = It->second;

  // Add the empty `ArrayPath` representing the case where no `array` is
  // traversed. This ensures that `toExplicitArrayAccesses` can produce
  // explicit `Arithmetic`s even when no array traversal is involved.
  ArrayPaths.push_back(ArrayPath());

  // Recursively traverse the `BaseType` to populate `Traversal`s and
  // `ArrayPath`s
  rc_eval(traverseImpl(BaseType, Traversals, ArrayPaths));

  // Sort traversals by `StartOffset`, then by `Size` of the `TargetType`
  std::sort(Traversals.begin(),
            Traversals.end(),
            [](const Traversal &A, const Traversal &B) {
              if (A.StartOffset != B.StartOffset) {
                return A.StartOffset < B.StartOffset;
              } else {
                auto ASize = getObjectSizeOrZero(A.TargetType);
                auto BSize = getObjectSizeOrZero(B.TargetType);
                return ASize < BSize;
              }
            });

  // Sort `ArrayPath`s by `Stride`, larger first
  for (auto &Path : ArrayPaths) {
    std::sort(Path.begin(),
              Path.end(),
              [](const NestedArrayShape &A, const NestedArrayShape &B) {
                return A.Stride > B.Stride;
              });
  }

  return It;
}

RecursiveCoroutine<void>
TypeTraversalAnalyzer::traverseImpl(mlir::Type Type,
                                    std::vector<Traversal> &Traversals,
                                    std::vector<ArrayPath> &ArrayPaths,
                                    int64_t CurrentOffset,
                                    const std::vector<uint64_t> &FieldPath,
                                    const ArrayPath &CurrentArrayPath) {

  // We should never reach a type with zero size - if we do, it means there is
  // something severely wrong in the types we're working with
  revng_assert(getObjectSizeOrZero(Type) > 0);

  if (auto PrimitiveType = mlir::dyn_cast<clift::PrimitiveType>(Type)) {
    // `PrimitiveType` is a leaf node in our traversal
    Traversals.emplace_back(PrimitiveType,
                            CurrentOffset,
                            0,
                            FieldPath,
                            CurrentArrayPath);
    rc_return;
  }

  // `PointerType` is a leaf node in our traversal: we do not traverse
  // through pointers, but we still want to produce a `Traversal` that
  // lands on a field whose type is a pointer
  if (auto Pointer = mlir::dyn_cast<clift::PointerType>(Type)) {
    Traversals.emplace_back(Pointer,
                            CurrentOffset,
                            0,
                            FieldPath,
                            CurrentArrayPath);
    rc_return;
  }

  // Traverse each `typedef`
  if (auto Typedef = mlir::dyn_cast<clift::TypedefType>(Type)) {
    Traversals.emplace_back(Typedef,
                            CurrentOffset,
                            0,
                            FieldPath,
                            CurrentArrayPath);
    rc_recur traverseImpl(Typedef.getUnderlyingType(),
                          Traversals,
                          ArrayPaths,
                          CurrentOffset,
                          FieldPath,
                          CurrentArrayPath);
    rc_return;
  }

  // Traverse the `array`
  if (auto ArrayType = mlir::dyn_cast<clift::ArrayType>(Type)) {
    Traversals.emplace_back(ArrayType,
                            CurrentOffset,
                            0,
                            FieldPath,
                            CurrentArrayPath);
    mlir::Type ElementType = ArrayType.getElementType();
    uint64_t NumElements = ArrayType.getElementsCount();
    uint64_t ElementSize = getObjectSize(ElementType);

    // Add this array to the current array path
    NestedArrayShape ArrayInfo;
    ArrayInfo.OffsetFromParentArrayElement = 0;
    ArrayInfo.Stride = ElementSize;
    ArrayInfo.NumElements = NumElements;

    ArrayPath NewArrayPath = CurrentArrayPath;
    if (!NewArrayPath.empty()) {

      // Adjust `OffsetFromParentArray` for nested arrays
      ArrayInfo
        .OffsetFromParentArrayElement = CurrentOffset
                                        - (NewArrayPath.back()
                                             .OffsetFromParentArrayElement);
    } else {
      ArrayInfo.OffsetFromParentArrayElement = CurrentOffset;
    }
    NewArrayPath.push_back(ArrayInfo);

    // Record this array path
    ArrayPaths.push_back(NewArrayPath);

    // Traverse into the first element of the array
    rc_recur traverseImpl(ElementType,
                          Traversals,
                          ArrayPaths,
                          CurrentOffset,
                          FieldPath,
                          NewArrayPath);
    rc_return;
  }

  // Traverse `struct` or `union` (both implement `ClassType`).
  // For `union`s, `Field.getOffset()` always returns 0 by verification.
  if (auto ClassType = mlir::dyn_cast<clift::ClassType>(Type)) {
    Traversals.emplace_back(ClassType,
                            CurrentOffset,
                            0,
                            FieldPath,
                            CurrentArrayPath);
    llvm::ArrayRef<clift::FieldAttr> Fields = ClassType.getFields();
    for (size_t I = 0; I < Fields.size(); ++I) {
      clift::FieldAttr Field = Fields[I];
      mlir::Type FieldType = Field.getType();
      int64_t FieldOffset = CurrentOffset + Field.getOffset();

      std::vector<uint64_t> NewFieldPath = FieldPath;
      NewFieldPath.push_back(static_cast<uint64_t>(I));

      rc_recur traverseImpl(FieldType,
                            Traversals,
                            ArrayPaths,
                            FieldOffset,
                            NewFieldPath,
                            CurrentArrayPath);
    }
    rc_return;
  }

  // Traverse the `enum`
  if (auto EnumType = mlir::dyn_cast<clift::EnumType>(Type)) {

    // We traverse the underlying `EnumType`
    mlir::Type UnderlyingType = EnumType.getUnderlyingType();

    // Add traversal for the `enum` itself
    Traversals.emplace_back(EnumType,
                            CurrentOffset,
                            0,
                            FieldPath,
                            CurrentArrayPath);

    // Also traverse into the underlying type inside the `enum
    rc_recur traverseImpl(UnderlyingType,
                          Traversals,
                          ArrayPaths,
                          CurrentOffset,
                          FieldPath,
                          CurrentArrayPath);
    rc_return;
  }
}

// =============================================================================
// `BestTraversalChooser` class definition
// =============================================================================

/// `BestTraversalChooser` is used as a compute class for the `BestTraversal`
class BestTraversalChooser {
private:
  /// The `TypeTraversalAnalyzer` is our `Traversal` and `ArrayPath` oracle
  TypeTraversalAnalyzer TraversalAnalyzer;

  /// Bit width of the pointer type being rewritten
  unsigned PointerBitWidth = 0;

public:
  /// We need an explicit constructor in order to propagate the
  /// `TraversalInfoMap` which is used as a global cache for storing
  /// `Traversal`s and `ArrayPath`s
  BestTraversalChooser(TraversalInfoMap &Data) : TraversalAnalyzer(Data) {}

public:
  /// Public entry point for computing the `BestTraversal` for the
  /// `PointerToReplace` and the pre-computed `Arithmetic`
  std::optional<Traversal>
  computeBestTraversal(ExpressionOpInterface PointerToReplace,
                       const PointerArithmetic &Arithmetic);

private:
  /// Obtain the explicit rewrite of the constant folded portion of
  /// `Arithmetic`, following an array traversal described by `ArrayPath`, so
  /// that it is evident in the `LinearCombination` component of `Arithmetic`
  std::optional<PointerArithmetic>
  getExplicitArithmetic(const PointerArithmetic &Arithmetic,
                        const ArrayPath &ArrayPath);

  /// Obtain all the explicit rewritings of the input `Arithmetic` following all
  /// the `ArrayPath`s for the `BaseType`. We explicitly specify the `BaseType`
  /// to handle the _pointer as array_ situation.
  std::vector<PointerArithmetic>
  toExplicitArrayAccesses(const PointerArithmetic &Arithmetic,
                          mlir::Type BaseType);

  /// Helper which trivially spill a `PointerArithmetic` into a `Traversal`
  Traversal toTraversal(const PointerArithmetic &PA,
                        mlir::Type PointeeType,
                        const Traversal &Ideal);

  /// Obtain the best `Traversal`
  std::optional<Traversal>
  getBestTraversal(mlir::Type BaseType,
                   mlir::Type PointeeType,
                   const std::vector<PointerArithmetic> &ExplicitArithmetics);
};

std::optional<Traversal>
BestTraversalChooser::computeBestTraversal(ExpressionOpInterface
                                             PointerToReplace,
                                           const PointerArithmetic
                                             &Arithmetic) {
  // We only perform the substitution for `PointerType`
  auto PointerToReplaceType = PointerToReplace->getResult(0).getType();
  if (not clift::unwrapped_isa<clift::PointerType>(PointerToReplaceType)) {
    return std::nullopt;
  }

  // Derive the `BaseType` for the traversal analysis, potentially wrapping the
  // `BasePointerType` into an implicit array (`p[i]` rewrite).
  auto BaseType = deriveBaseType(Arithmetic.BasePointer);

  // It may be that the `PointerToReplace` points to a `void 0` type, in that
  // case we cannot provide a `Traversal` for sure
  if (getObjectSizeOrZero(BaseType) == 0) {
    return std::nullopt;
  }

  // Expand to explicit array accesses the input `PointerArithmetic`, so that
  // the constant folded component performed by the compiler is evident in the
  // `LinearCombination` portion of `Arithmetic`
  PointerBitWidth = clift::unwrapped_cast<PointerType>(PointerToReplaceType)
                      .getPointerSize()
                    * 8;

  std::vector<PointerArithmetic>
    ExplicitArithmetics = toExplicitArrayAccesses(Arithmetic, BaseType);

  mlir::Type
    PointeeType = clift::unwrapped_cast<PointerType>(PointerToReplaceType)
                    .getPointeeType();

  // Obtain the `BestTraversal` for connecting `BaseType` to `PointeeType`,
  // following one of the possible `ExplicitArithmetic`s
  auto BestTraversal = getBestTraversal(BaseType,
                                        PointeeType,
                                        ExplicitArithmetics);

  // In case we end up with a `BestTraversal` which does not actually traverse
  // any `struct` field or `array` element, we avoid the rewriting altogether,
  // and we leave the explicit pointer arithmetic access in `clift`
  if (BestTraversal->depth() == 0) {
    return std::nullopt;
  }

  return BestTraversal;
}

// Turn the input `Arithmetic` into another `PointerArithmetic` , following the
// array traversal dictaded by the `ArrayPath` `AP`. On success, the output
// `PointerArithmetic` has a `StartOffset` that is lower than or equal than the
// input one, and all the array traversals of `AP` have an equivalent
// `StridedTerm` in the `LinearCombination` of `OffsetExpression` of the
// `Result`.
// For example this can turn P + 12 (with no `LinearCombination`) into
// P + 8 * 1 + 4 (with a single `StridedTerm`, with fixed `Index`) if `AP` is
// e.g. `{.OffsetFromParentArrayElement = 0, .EndOffset = 48, .Stride = 8}`)
std::optional<PointerArithmetic>
BestTraversalChooser::getExplicitArithmetic(const PointerArithmetic &Arithmetic,
                                            const ArrayPath &AP) {

  PointerArithmetic Result{
    .BasePointer = Arithmetic.BasePointer,
    .Offset = PointerArithmetic::OffsetExpression(PointerBitWidth)
  };

  // We do not modify the input `Arithmetic`, but we work on a local copy
  PointerArithmetic WorkingArithmetic = Arithmetic;

  for (const NestedArrayShape &NAI : AP) {
    const auto &[OffsetFromParentArrayElement, NumElements, Stride] = NAI;

    // Consume the offset from the parent array element
    Result.Offset.BaseOffset += OffsetFromParentArrayElement;

    revng_assert(WorkingArithmetic.Offset.BaseOffset
                   .uge(OffsetFromParentArrayElement));
    llvm::APInt OffsetInsideArray = WorkingArithmetic.Offset.BaseOffset
                                    - OffsetFromParentArrayElement;
    WorkingArithmetic.Offset.BaseOffset = OffsetInsideArray;

    // If, after adjusting `StartOffset`, the `StartOffset` of `Arithmetic` is
    // still larger than the `Stride` coming from `NAI`, we have to take that
    // into account and add to the `Result` a constant `StridedTerm`
    auto &LC = Result.Offset.LinearCombination;

    // `Stride`s must be in non-ascending order (equal strides are allowed for
    // nested arrays with the same element size, e.g., int array[1][1])
    revng_assert(LC.empty() or LC.back().Stride.uge(Stride));

    llvm::APInt IndexConstantComponent = llvm::APInt(PointerBitWidth, 0);
    if (WorkingArithmetic.Offset.BaseOffset.uge(Stride)) {
      IndexConstantComponent = WorkingArithmetic.Offset.BaseOffset.udiv(Stride);
      revng_assert(IndexConstantComponent.ult(NumElements));
      WorkingArithmetic.Offset.BaseOffset = WorkingArithmetic.Offset.BaseOffset
                                              .urem(Stride);
    }

    mlir::Value IndexVariableComponent = {};

    // If the `PointerArithmetic` is doing a variable-index array traversal with
    // a stride that differs from NAI, then we have to bail out. Either a
    // subsequent `ArrayPath` will hit the sweet spot, or none will, but that's
    // not something we have to handle here.
    if (not WorkingArithmetic.Offset.LinearCombination.empty()) {
      if (WorkingArithmetic.Offset.LinearCombination.front()
            .Stride.ugt(Stride)) {
        return std::nullopt;
      }
      if (Stride == WorkingArithmetic.Offset.LinearCombination.front().Stride) {
        auto &FrontTerm = WorkingArithmetic.Offset.LinearCombination.front();
        IndexVariableComponent = FrontTerm.Idx.Variable;

        // If the index also has a constant component, sum it into the
        // `IndexConstantComponent` we are building
        if (FrontTerm.Idx.Constant.getBoolValue()) {
          IndexConstantComponent += FrontTerm.Idx.Constant;
        }

        WorkingArithmetic.Offset.LinearCombination
          .erase(WorkingArithmetic.Offset.LinearCombination.begin());
      }
    }

    // We add a new term to the constructed `LinearCombination`, using the
    // `Index` components identified
    LC.push_back(PointerArithmetic::StridedTerm(llvm::APInt(PointerBitWidth,
                                                            Stride),
                                                { IndexVariableComponent,
                                                  IndexConstantComponent }));
  }

  // If we were not able to consume all the `LinerCombination`s of the input
  // `Arithmetic`, we bail out
  if (not WorkingArithmetic.Offset.LinearCombination.empty()) {
    return std::nullopt;
  }

  // If we still have some non-consumed portion of the input `BaseOffset`, we
  // propagate it in the `Result`
  if (WorkingArithmetic.Offset.BaseOffset.getBoolValue()) {
    Result.Offset.BaseOffset += WorkingArithmetic.Offset.BaseOffset;
  }

  return Result;
}

std::vector<PointerArithmetic>
BestTraversalChooser::toExplicitArrayAccesses(const PointerArithmetic
                                                &Arithmetic,
                                              mlir::Type BaseType) {
  std::vector<PointerArithmetic> Result;

  // We retrieve all the `ArrayPath`s that we can build from `BaseType`
  const std::vector<ArrayPath> &ArrayPaths = TraversalAnalyzer
                                               .getArrayPaths(BaseType);

  // We now filter the `ArrayPath`s by taking into consideration only those that
  // are compatible with the `BaseOffset` access present in the
  // `PointerArithmetic` that we are considering
  auto CompatibleArrayPaths = findCompatibleArrayPaths(ArrayPaths,
                                                       Arithmetic.Offset
                                                         .BaseOffset);

  for (const ArrayPath *TheArrayPath : CompatibleArrayPaths) {

    // Try and turn `Arithmetic` into a form where all array indexes, even the
    // constant ones, are explicit. In case of success, we enqueue the
    // `Explicit` in the final `Result`s
    std::optional<PointerArithmetic>
      Explicit = getExplicitArithmetic(Arithmetic, *TheArrayPath);
    if (Explicit) {
      Result.push_back(std::move(*Explicit));
    }
  }

  return Result;
}

Traversal BestTraversalChooser::toTraversal(const PointerArithmetic &PA,
                                            mlir::Type PointeeType,
                                            const Traversal &Ideal) {

  // Ensure that we never end up with a negative `LeftoverOffset`, which would
  // not make any sense
  revng_assert(Ideal.StartOffset >= 0);
  revng_assert(PA.Offset.BaseOffset.uge(Ideal.StartOffset));

  // We want to turn an explicit `PointerArithmetic` `PA` in a `Traversal`,
  // assuming that it does the traversal described in `Ideal`.
  // Because `PA` is explicit (i.e. all array traversals at fixed index have
  // been expanded in the `LinearCombination`), the `Result` `Traversal` will
  // always be the same, except that we have to adjust the leftover offset
  // and fix the `TargetType` to the actual type required by the traversal
  auto BaseOffset = PA.Offset.BaseOffset.getZExtValue();
  Traversal Result = Ideal;
  Result.TargetType = PointeeType;
  Result.LeftoverOffset = BaseOffset - Ideal.StartOffset;
  return Result;
}

// To find the `BestTraversal`, we have to compare all the valid `Traversal`s
// from `BaseType` with all the ones that we can build from the
// `ExplicitArithmetic`s we get from `clift`.
std::optional<Traversal>
BestTraversalChooser::getBestTraversal(mlir::Type BaseType,
                                       mlir::Type PointeeType,
                                       const std::vector<PointerArithmetic>
                                         &ExplicitArithmetics) {

  std::optional<Traversal> BestTraversal = std::nullopt;
  Score BestScore = Score::invalid();

  // The `Traversal`s are lazily computed upon first inspection of a `BaseType`
  const std::vector<Traversal> &Traversals = TraversalAnalyzer
                                               .getTraversals(BaseType);

  for (const PointerArithmetic &Explicit : ExplicitArithmetics) {

    // Get the range of `Traversal`s to compare from the `TraversalAnalyzer`.
    // There are two modes of operation for this: with `SmartLookup`, or
    // without.
    // ATM only `SmartLookup` is not implemented. Once it is however,
    // in the tests we want to double check that the results obtained are the
    // SAME both if `SmartLookup` enabled and disabled.
    auto [Begin, End] = TraversalAnalyzer.getTraversalRange(BaseType,
                                                            Explicit,
                                                            PointeeType,
                                                            false);
    for (auto It = Begin; It != End; ++It) {
      const Traversal &Ideal = *It;

      // Skip `Ideals` whose `StartOffset` exceeds the `Explicit` `BaseOffset`,
      // the `Explicit` cannot reach that far, and `toTraversal` would
      // underflow.
      auto BaseOffset = Explicit.Offset.BaseOffset.getZExtValue();
      if (Explicit.Offset.BaseOffset.ult(Ideal.StartOffset))
        continue;

      // Convert each `ExplicitArithemtic` into a `Traversal`, so we can compare
      // it with `Ideal`. `ExplicitTraversal` is the `Traversal` that we
      // would obtain traversing the `BaseType` with `Explicit` if we did
      // traverse it as the `Ideal` suggests. Basically what can be
      // different is just the `LeftOverOffset`.
      Traversal ExplicitTraversal = toTraversal(Explicit, PointeeType, Ideal);

      Score CurrentScore = score(ExplicitTraversal, Ideal);

      if (!CurrentScore.Valid)
        continue;

      // We select the `Score` which best suits the criteria defining in the
      // _scoring_ mechanism
      if (CurrentScore < BestScore) {
        BestScore = CurrentScore;
        BestTraversal = ExplicitTraversal;
      }
    }
  }

  // We serialize on the `Log` the selected `BestTraversal`
  if (Log.isEnabled() and BestTraversal) {
    Log << "Elected  BestTraversal:\n";
    BestTraversal->dump();
  }

  // If no `Traversal` was valid, we'll return a `nullopt` here, meaning that we
  // will not replace the `PointerToReplace` with a field access. This basically
  // means that all the possible `Traversal`s that the clift expression could
  // represent are so ugly that we bail out.
  return BestTraversal;
}

} // namespace

std::optional<Traversal>
computeBestTraversal(ExpressionOpInterface PointerToReplace,
                     const PointerArithmetic &Arithmetic,
                     TraversalInfoMap &Data) {
  auto BestTraversalC = BestTraversalChooser(Data);
  return BestTraversalC.computeBestTraversal(PointerToReplace, Arithmetic);
}
