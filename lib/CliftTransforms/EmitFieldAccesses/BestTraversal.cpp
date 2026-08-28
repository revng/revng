//
// This file is distributed under the MIT License. See LICENSE.md for details.
//
#include <compare>
#include <cstdint>
#include <limits>
#include <optional>

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/MathExtras.h"

#include "revng/ADT/RecursiveCoroutine.h"
#include "revng/Clift/Clift.h"
#include "revng/Support/Assert.h"
#include "revng/Support/CommandLine.h"

#include "BestTraversal.h"
#include "PointerArithmetic.h"

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

  // If the pointee is a struct, union, or array, use it directly — the
  // traversal analyzer can walk its fields and arrays.
  // Wrapping also a `struct`, would mean rewriting a constant offset access
  // into it as `p[0].field` instead of `p->field`.
  if (unwrapped_isa<StructType, UnionType, ArrayType>(PointeeType)) {
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

/// Checks whether the arrays traversed by `T` can represent every runtime
/// (variable-index) strided term of `Arithmetic`. `Replacement::make` lowers a
/// variable strided term into an array subscript when a traversed array's
/// stride divides the term's stride (the exact-stride match is the divisor
/// being the stride itself; a proper multiple yields a scaled index such as
/// `[2*i]` into a byte array), so a term is representable exactly when some
/// traversed array's stride divides it. A variable term whose stride is a
/// multiple of no traversed array's stride cannot be an integer array index
/// and, carrying a runtime index, would be silently dropped by the leftover
/// emission in `replaceFieldAccess`. A traversal that fails this check is not a
/// faithful lowering of the access, so `getBestTraversal` never selects it as
/// the best; a representable, lower-scored traversal is preferred, and only
/// when none exists is the raw pointer arithmetic kept (as for accesses that
/// fail `PointerArithmetic::verify`). Constant-only leftover terms are fine:
/// they are emitted as explicit integer arithmetic.
static bool canLowerArithmeticOntoTraversal(const PointerArithmetic &Arithmetic,
                                            const Traversal &T) {
  for (const auto &Term : Arithmetic.Offset.LinearCombination) {
    if (not Term.Idx.Variable)
      continue;

    bool Divisible = false;
    for (const ArrayShape &Array : T.TraversedArrays) {
      if (Term.Stride.urem(Array.Stride) == 0) {
        Divisible = true;
        break;
      }
    }

    if (not Divisible)
      return false;
  }

  return true;
}

/// The `typeDistance` function computes the distance between two `Type`s.
/// Returns 0 if the types are exactly equal (after stripping `typedef`s), 1 if
/// they only differ in CV-qualifiers, or infinity otherwise. The return type is
/// kept as `uint64_t` to allow future refinement of the scoring criterion,
/// enabling more fine grained control over this score.
static uint64_t typeDistance(mlir::Type Explicit, mlir::Type Ideal) {
  if (unwrapTypedefs(Explicit) == unwrapTypedefs(Ideal)) {
    return 0;
  }

  if (equivalentUnwrapped(Explicit, Ideal)) {
    return 1;
  }

  return std::numeric_limits<uint64_t>::max();
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
/// pointer access rewrite.
///
/// `AccessArrays` are the array shapes the access actually walks (the shapes of
/// the `ArrayPath` the `Explicit` was built from). They are compared against
/// `Ideal.TraversedArrays` to compute the `CommonStrides` criterion. They are
/// passed separately, rather than read from `Explicit.TraversedArrays`, because
/// `toTraversal` builds the `Explicit` `Traversal` as a copy of the `Ideal`,
/// so its `TraversedArrays` would be identical to `Ideal.TraversedArrays` and
/// `CommonStrides` would degenerate to the `Ideal`'s array count.
static Score score(const Traversal &Explicit,
                   const Traversal &Ideal,
                   llvm::ArrayRef<ArrayShape> AccessArrays) {
  int64_t StartDistance = Explicit.begin() - Ideal.begin();
  int64_t EndDistance = Explicit.end() - Ideal.end();

  uint64_t CommonStrides = commonPrefixStrides(AccessArrays,
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

static const char *D = "Upper bound on the number of type traversals "
                       "emit-field-accesses materializes per type before "
                       "leaving an access as raw pointer arithmetic; "
                       "0 removes the bound.";

/// Upper bound on the number of `Traversal`s that `TypeTraversalAnalyzer` will
/// materialize for a single `BaseType`. Real, non-degenerate types stay far
/// below this; degenerate DLA type systems, whose deeply-nested high-arity
/// unions share substructure, reach millions of distinct root-to-leaf
/// traversals. Above this bound EFA bails out of the rewrite for accesses into
/// the offending type (leaving the raw pointer arithmetic), which caps both the
/// memory used to store the traversals and the time spent scoring an access
/// against them. A value of 0 removes the bound, treating it as infinite.
static llvm::cl::opt<uint64_t> MaxTraversalsPerType("max-traversals-per-type",
                                                    llvm::cl::init(1024),
                                                    llvm::cl::cat(MainCategory),
                                                    llvm::cl::desc(D));

/// `TypeTraversalAnalyzer` is used as an oracle to compute and retrieve
/// `Traversal`s and `ArrayPath`s in a lazy manner
class TypeTraversalAnalyzer {
private:
  /// Reference to the cache containing the pre-computed `Traversal`s and
  /// `ArrayPath`s. This passed in the constructor, as we want to cache the
  /// information across different runs
  TraversalInfoMap &Data;

  /// Memoizes `countTraversals`. A type's traversal count depends only on the
  /// type, so it is safe to cache and reuse across `BaseType`s.
  llvm::DenseMap<mlir::Type, uint64_t> TraversalCountCache;

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
  /// Counts, without materializing them, how many `Traversal`s `traverseImpl`
  /// would emit for `Type` (one per traversed node; pointers are leaves).
  /// Saturates at `MaxTraversalsPerType + 1`, so a degenerate, deeply-shared
  /// type cannot overflow the counter or make counting itself expensive. Uses
  /// `RecursiveCoroutine` for stack safety on deeply nested types (like
  /// `traverseImpl`); evaluate it with `rc_eval`.
  RecursiveCoroutine<uint64_t> countTraversals(mlir::Type Type);

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

RecursiveCoroutine<uint64_t>
TypeTraversalAnalyzer::countTraversals(mlir::Type Type) {
  // A type's traversal count depends only on the type: reuse cached results.
  if (auto It = TraversalCountCache.find(Type); It != TraversalCountCache.end())
    rc_return It->second;

  // By-value types cannot form a cycle (it would have infinite size), but seed
  // the cache with a saturated value before recurring so that, if a malformed
  // type ever did, it would be treated as over the cap rather than recurring
  // forever. `llvm::SaturatingAdd` likewise keeps the running count from
  // overflowing on a degenerate type; only its comparison with the cap matters.
  TraversalCountCache[Type] = std::numeric_limits<uint64_t>::max();

  uint64_t Count = 0;
  if (mlir::isa<clift::PrimitiveType, clift::PointerType>(Type)) {
    // Leaves: `traverseImpl` emits a single `Traversal` and stops (in
    // particular it does not follow pointers).
    Count = 1;
  } else if (auto Typedef = mlir::dyn_cast<clift::TypedefType>(Type)) {
    uint64_t Underlying = rc_recur countTraversals(Typedef.getUnderlyingType());
    Count = llvm::SaturatingAdd(Underlying, uint64_t{ 1 });
  } else if (auto Array = mlir::dyn_cast<clift::ArrayType>(Type)) {
    uint64_t Element = rc_recur countTraversals(Array.getElementType());
    Count = llvm::SaturatingAdd(Element, uint64_t{ 1 });
  } else if (auto Class = mlir::dyn_cast<clift::ClassType>(Type)) {
    // One `Traversal` for the class itself, plus every field's subtree.
    Count = 1;
    for (clift::FieldAttr Field : Class.getFields()) {
      uint64_t FieldCount = rc_recur countTraversals(Field.getType());
      Count = llvm::SaturatingAdd(Count, FieldCount);
    }
  } else if (auto Enum = mlir::dyn_cast<clift::EnumType>(Type)) {
    uint64_t Underlying = rc_recur countTraversals(Enum.getUnderlyingType());
    Count = llvm::SaturatingAdd(Underlying, uint64_t{ 1 });
  }

  TraversalCountCache[Type] = Count;
  rc_return Count;
}

llvm::DenseMap<mlir::Type, TraversalInfo>::iterator
TypeTraversalAnalyzer::traverse(mlir::Type BaseType) {
  revng_assert(Data.count(BaseType) == 0);

  auto [It, Inserted] = Data.insert({ BaseType, TraversalInfo() });
  auto &[Traversals, ArrayPaths] = It->second;

  // Some DLA type systems contain deeply-nested, high-arity unions whose fields
  // share substructure, so that the number of distinct root-to-leaf traversals
  // is exponential in the nesting depth (millions, even for types with only a
  // handful of fields each). Enumerating and storing them all exhausts memory,
  // and scoring an access against all of them does not terminate in reasonable
  // time. Counting the traversals is cheap (memoized, one value per type) where
  // enumerating them is not, so we count first and, above the cap, bail out
  // leaving `Traversals`/`ArrayPaths` empty. Downstream, `computeBestTraversal`
  // then finds no traversal and EFA leaves the raw pointer arithmetic in place,
  // exactly as it does for any access it cannot lower onto a field access.
  if (MaxTraversalsPerType != 0
      and rc_eval(countTraversals(BaseType)) > MaxTraversalsPerType) {
    llvm::StringRef Handle = "<anonymous>";
    if (auto Defined = mlir::dyn_cast<clift::DefinedType>(BaseType))
      Handle = Defined.getHandle();
    revng_log(Log,
              "WARNING: skipping field-access rewrite for type '"
                << Handle << "' producing more than "
                << MaxTraversalsPerType.getValue() << " traversals");
    return It;
  }

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
// `ExplicitPointerArithmetic` struct definition
// =============================================================================

/// An `ExplicitPointerArithmetic` pairs an explicit `PointerArithmetic` (a
/// `PointerArithmetic` re-expressed along a concrete `ArrayPath`, so that every
/// array index is evident in its `LinearCombination`) with the array shapes of
/// that `ArrayPath`, sorted in the same descending-stride order as
/// `Traversal::TraversedArrays`.
///
/// `AccessArrays` records the array structure the access *actually* walks. It
/// is what the `CommonStrides` scoring criterion compares against each
/// candidate `Ideal` traversal's arrays: an access stepping through a stride-2
/// array should prefer an `Ideal` that goes through a stride-2 array. This is
/// kept separate from the `Traversal` produced by `toTraversal`, whose
/// `TraversedArrays` are a copy of the `Ideal`'s (used, unchanged, to drive the
/// emission and the runtime-index representability checks).
struct ExplicitPointerArithmetic {
  PointerArithmetic Arithmetic;
  llvm::SmallVector<ArrayShape> AccessArrays;
};

/// A list of `ExplicitPointerArithmetic`s, one per compatible `ArrayPath`
using ExplicitPointerArithmetics = std::vector<ExplicitPointerArithmetic>;

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
  /// the `ArrayPath`s for the `BaseType`. Each rewriting is paired with the
  /// array shapes of the `ArrayPath` it followed (see
  /// `ExplicitPointerArithmetic`). We explicitly specify the `BaseType` to
  /// handle the _pointer as array_ situation.
  ExplicitPointerArithmetics
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
                   const PointerArithmetic &Arithmetic,
                   const ExplicitPointerArithmetics &ExplicitAccesses);
};

std::optional<Traversal>
BestTraversalChooser::computeBestTraversal(ExpressionOpInterface
                                             PointerToReplace,
                                           const PointerArithmetic
                                             &Arithmetic) {
  // We only perform the substitution for `PointerType`
  auto PointerToReplaceType = PointerToReplace.getType();
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

  ExplicitPointerArithmetics
    ExplicitAccesses = toExplicitArrayAccesses(Arithmetic, BaseType);

  mlir::Type
    PointeeType = clift::unwrapped_cast<PointerType>(PointerToReplaceType)
                    .getPointeeType();

  // Obtain the `BestTraversal` for connecting `BaseType` to `PointeeType`,
  // following one of the possible `ExplicitArithmetic`s. Traversals that cannot
  // represent the arithmetic's runtime indices are excluded during selection
  // (see `getBestTraversal`), so if none remains we leave the raw pointer
  // arithmetic in place.
  auto BestTraversal = getBestTraversal(BaseType,
                                        PointeeType,
                                        Arithmetic,
                                        ExplicitAccesses);

  // In case we end up with a `BestTraversal` which does not actually traverse
  // any `struct` field or `array` element, we avoid the rewriting altogether,
  // and we leave the explicit pointer arithmetic access in `clift`
  if (not BestTraversal or BestTraversal->depth() == 0) {
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

  // The variable strided terms are runtime indices that making the array
  // indexing explicit does not change: a term's stride is a multiple of the
  // element stride of the array that indexes it, which already encodes its
  // coefficient. We therefore keep them in the `Result` unchanged and only
  // split the constant part of the offset into an explicit constant index per
  // array level. `WorkingArithmetic` is consumed purely to check that every
  // variable term is representable (divisible by some array in the path).
  auto &LC = Result.Offset.LinearCombination;
  LC = Arithmetic.Offset.LinearCombination;

  for (const NestedArrayShape &NAI : AP) {
    const auto &[OffsetFromParentArrayElement, NumElements, Stride] = NAI;

    // Consume the offset from the parent array element
    Result.Offset.BaseOffset += OffsetFromParentArrayElement;

    revng_assert(WorkingArithmetic.Offset.BaseOffset
                   .uge(OffsetFromParentArrayElement));
    WorkingArithmetic.Offset.BaseOffset -= OffsetFromParentArrayElement;

    // Split the constant offset that reaches into this array into an explicit
    // constant array index. We fold it into the term that already indexes this
    // array, if any (i.e. a variable term of the same stride), or add a new
    // constant-index term.
    if (WorkingArithmetic.Offset.BaseOffset.uge(Stride)) {
      llvm::APInt FixedIndex = WorkingArithmetic.Offset.BaseOffset.udiv(Stride);
      revng_assert(FixedIndex.ult(NumElements));
      WorkingArithmetic.Offset.BaseOffset = WorkingArithmetic.Offset.BaseOffset
                                              .urem(Stride);

      auto SameStride = std::find_if(LC.begin(),
                                     LC.end(),
                                     [&](const PointerArithmetic::StridedTerm
                                           &Term) {
                                       return Term.Stride == Stride;
                                     });
      if (SameStride != LC.end())
        SameStride->Idx.Constant += FixedIndex;
      else
        LC.emplace_back(llvm::APInt(PointerBitWidth, Stride),
                        PointerArithmetic::Index{ mlir::Value(), FixedIndex });
    }

    // Verify this path can represent every runtime index: remove from the
    // working copy every term this array's stride divides (an exact match, or a
    // proper multiple such as a two-byte-stride index into a byte array). A
    // term this array does not divide is left for a later, smaller-stride
    // array; if no array consumes it, the check after the loop discards this
    // path.
    auto &WorkingLC = WorkingArithmetic.Offset.LinearCombination;
    for (auto It = WorkingLC.begin(); It != WorkingLC.end();) {
      if (It->Stride.urem(Stride) == 0)
        It = WorkingLC.erase(It);
      else
        ++It;
    }
  }

  // If some runtime index is a multiple of no array's stride, this path cannot
  // represent it, so we bail out
  if (not WorkingArithmetic.Offset.LinearCombination.empty()) {
    return std::nullopt;
  }

  // If we still have some non-consumed portion of the input `BaseOffset`, it
  // lands inside the target element, so we propagate it in the `Result`
  if (WorkingArithmetic.Offset.BaseOffset.getBoolValue()) {
    Result.Offset.BaseOffset += WorkingArithmetic.Offset.BaseOffset;
  }

  // Restore descending-stride order, which the construction above does not
  // guarantee: the constant array-index terms are appended after the copied
  // (already-ordered) runtime terms, so a large-stride outer array index can
  // land after a smaller-stride runtime term (e.g. `arr[c][i]`, offset
  // `8*c + 4*i`, produces `[{4, i}, {8, c}]`). No current consumer depends on
  // the order (the only reader, the `canLowerArithmeticOntoTraversal` assertion
  // in `getBestTraversal`, is order-independent), but we keep the result a
  // well-formed `PointerArithmetic`, matching the descending, unique-stride
  // invariant that `computePointerArithmetic` and `verify` maintain, so the
  // object stays consistent and safe for any future consumer.
  std::sort(LC.begin(),
            LC.end(),
            [](const PointerArithmetic::StridedTerm &A,
               const PointerArithmetic::StridedTerm &B) {
              return A.Stride.ugt(B.Stride);
            });

  return Result;
}

ExplicitPointerArithmetics
BestTraversalChooser::toExplicitArrayAccesses(const PointerArithmetic
                                                &Arithmetic,
                                              mlir::Type BaseType) {
  ExplicitPointerArithmetics Result;

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
    // `Explicit`, together with the array shapes it walked, in the final
    // `Result`s. The array shapes carry the access's real array structure,
    // which the `CommonStrides` criterion compares against each candidate
    // `Ideal`.
    std::optional<PointerArithmetic>
      Explicit = getExplicitArithmetic(Arithmetic, *TheArrayPath);
    if (Explicit) {
      Result
        .push_back({ .Arithmetic = std::move(*Explicit),
                     .AccessArrays = arrayPathToSortedVector(*TheArrayPath) });
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
  // and fix the `TargetType` to the actual type required by the traversal.
  //
  // Note the `Result`'s `TraversedArrays` deliberately stay those of the
  // `Ideal`: they describe the concrete type structure the emission walks (and
  // whose real strides the runtime-index representability check relies on), not
  // the array structure the access walks. The latter is carried separately in
  // `ExplicitPointerArithmetic::AccessArrays` and only used to score
  // `CommonStrides`.
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
                                       const PointerArithmetic &Arithmetic,
                                       const ExplicitPointerArithmetics
                                         &ExplicitAccesses) {

  std::optional<Traversal> BestTraversal = std::nullopt;
  Score BestScore = Score::invalid();

  // The `Traversal`s are lazily computed upon first inspection of a `BaseType`
  const std::vector<Traversal> &Traversals = TraversalAnalyzer
                                               .getTraversals(BaseType);

  for (const ExplicitPointerArithmetic &ExplicitAccess : ExplicitAccesses) {
    const PointerArithmetic &Explicit = ExplicitAccess.Arithmetic;

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

      Score CurrentScore = score(ExplicitTraversal,
                                 Ideal,
                                 ExplicitAccess.AccessArrays);

      if (!CurrentScore.Valid)
        continue;

      // A traversal whose array strides cannot represent the arithmetic's
      // runtime indices is not a valid lowering (see
      // `canLowerArithmeticOntoTraversal`), so it must never be considered
      // best: skip it here and let a representable, lower-scored traversal win.
      if (not canLowerArithmeticOntoTraversal(Arithmetic, ExplicitTraversal))
        continue;

      // The explicit arithmetic must be lowerable onto the very same arrays the
      // original arithmetic just was: `getExplicitArithmetic` re-expresses the
      // access along an `ArrayPath`, and its `LinearCombination` must keep
      // every runtime term at a stride the traversed arrays can represent. This
      // is the one live consumer of that `LinearCombination`, and it guards
      // `getExplicitArithmetic` against re-expressing a runtime index at a
      // stride no array divides.
      revng_assert(canLowerArithmeticOntoTraversal(Explicit,
                                                   ExplicitTraversal));

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
