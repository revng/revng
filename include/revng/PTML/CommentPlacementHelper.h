#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/ADT/DepthFirstIterator.h"
#include "llvm/ADT/SmallBitVector.h"
#include "llvm/IR/Dominators.h"
#include "llvm/Support/GenericDomTree.h"

#include "revng/ADT/RecursiveCoroutine.h"
#include "revng/Model/Function.h"
#include "revng/PTML/Doxygen.h"

namespace yield {

template<typename NodeType>
struct StatementTraits {
  // See `HasStatementTraits` for the list of what this trait should provide.
};

template<typename NodeType>
concept HasStatementTraits = requires(NodeType Node,
                                      StatementTraits<NodeType>::StatementType
                                        Statement) {
  // using StatementType = /* your type */
  typename StatementTraits<NodeType>::StatementType;

  // static RangeOf<StatementType> auto getStatements(NodeType);
  {
    StatementTraits<NodeType>::getStatements(Node)
  } -> RangeOf<typename StatementTraits<NodeType>::StatementType>;

  // static RangeOf<MetaAddress> auto getAddresses(StatementType);
  {
    StatementTraits<NodeType>::getAddresses(Statement)
  } -> RangeOf<MetaAddress>;
};

} // namespace yield

namespace llvm {

// These are necessary for llvm graph helpers (such as `llvm::depth_first`)
// to be usable on dominator trees.
//
// Luckily, we don't need to implement them from scratch, since
// `llvm::DomTreeNodeBase` already does so (under a different name),
// we just need to point in the right direction.

template<typename NodeType>
  requires yield::HasStatementTraits<NodeType *>
struct GraphTraits<llvm::DomTreeNodeBase<NodeType> *>
  : public DomTreeGraphTraitsBase<
      llvm::DomTreeNodeBase<NodeType>,
      typename llvm::DomTreeNodeBase<NodeType>::const_iterator> {};

template<typename NodeType>
  requires yield::HasStatementTraits<NodeType *>
struct GraphTraits<const llvm::DomTreeNodeBase<NodeType> *>
  : public DomTreeGraphTraitsBase<
      const llvm::DomTreeNodeBase<NodeType>,
      typename llvm::DomTreeNodeBase<NodeType>::const_iterator> {};

} // namespace llvm

namespace yield {

/// This is a helper for deciding which statement (within a given graph) is
/// best suited for emission of a given comment.
///
/// \tparam NodeType the type of the graph node.
///         Note that it has to specialize `StatementTraits`.
///
/// A typical usage would be something like:
///
/// ```cpp
/// // 1. Make a helper object from your function and your graph:
/// CommentPlacementHelper<MyNodeType> CommentPlacement(MyModelFunction,
///                                                     MyGraph);
/// // note: We need to be able to compute the dominator tree. In order to allow
/// //       that, you need to provide a `llvm::DomTreeNodeTraits<MyNodeType>`
/// //       trait OR `MyNodeType::getParent()` method.
///
/// // 2. When emitting your artifact, add calls to `getComments`:
/// for (const auto &Block : myWayOfIteratingTheGraph(MyGraph)) {
///   for (const auto &Statement : myWayOfGettingTheStatements(Block))
///     // . . .
///     for (const auto &Comment : CommentPlacement.getComments(Statement)) {
///       emitTheComment(Comment);
///       // note: see `PTML/Doxygen` for the standard way of emitting comments
///     }
///     // . . .
///   }
/// }
///
/// // 3. Don't forget to emit homeless comments somewhere.
/// //    These are the comments that do not have even a single match among
/// //    all the statements in the provided graph.
/// for (const auto &Comment : CommentPlacement.getHomelessComments())
///   emitTheComment(Comment);
/// ```
///
/// With the interface out of the way, lets now discuss *how* statements are
/// selected:
///
/// 1. For each comment-statement pair a similarity score is computed
///    (see \ref tverskyIndex) based on the similarity of the set of addresses
///    recorded in the comment to the set of addresses associated to
///    the statement.
/// 2. Then, for each comment a single *address set* (note: not a statement yet)
///    is selected.
/// 3. Then graph is traversed again, this time looking for all the statements
///    whose location exactly matches the selected one.
///    When there's an exact match, the comment is assigned to that statement.
///    This happens *unless* the current statement is dominated by another
///    statement that is already associated to the comment currently being
///    considered. This prevents emitting the same comment multiple times
///    in cases where it's obviously not beneficial.
/// 4. All these statements are recorded in a \ref ResultMap and will be emitted
///    once requested by a \ref getComments call.
///
/// For some specific examples on how this helper works, see the corresponding
/// unit tests, but let me also provide a trivial example. Let's say we have
/// a *really* sequential graph with three nodes: `(A) -> (B) -> (C)`.
/// And let's say each node has a single statement with addresses attached:
///   - A: { 0x001:Generic64 }
///   - B: { 0x001:Generic64, 0x002:Generic64 }
///   - C: { 0x001:Generic64, 0x002:Generic64 }
///
/// And let's say we're trying to decide where to emit a comment with
/// `{ 0x001:Generic64, 0x002:Generic64, 0x003:Generic64 }`
///
/// After running step 1 (above) on it, the resulting scores for locations(!)
/// are as follows (note that there are only two locations):
/// - `{ 0x001:Generic64 }`: 1/3
/// - `{ 0x001:Generic64, 0x002:Generic64 }`: 2/3
///
/// So the best one is selected. And there are two statements that correspond
/// to it: `B` and `C`.
///
/// If these two statement were not related, the comment would end up duplicated
/// BUT based on the graph, the only way to reach `C` is through `B`! As such,
/// there's not much downside to suppressing the `C` comment and only emitting
/// one before `B`.
template<HasStatementTraits NodeType>
class CommentPlacementHelper {
private:
  using Trait = StatementTraits<NodeType>;
  using InternalNodeType = std::remove_pointer_t<NodeType>;

  using StatementType = typename Trait::StatementType;
  using SLT = decltype(Trait::getAddresses(std::declval<StatementType>()));
  using StatementLocationType = std::decay_t<SLT>;

  struct Score {
    uint64_t Numerator = 0;
    uint64_t Denominator = 1;

    std::strong_ordering operator<=>(const Score &Another) const {
      // This takes advantage of the easiest way to compare two fractions:
      // instead of looking for the *smallest* common denominator,
      // just use *one* common denominator: product of the two denominators.
      //
      // Say, we're comparing `a/b` and `c/d`: *one* common denominator is
      // `b*d`, which means we can just compare `(a*d)/(b*d)` and `(b*c)/(b*d)`.
      // And since "adjusted" denominators are the same, we can also omit them
      // and just compare the numerators.
      uint64_t AdjustedLHS = this->Numerator * Another.Denominator;
      uint64_t AdjustedRHS = Another.Numerator * this->Denominator;
      return AdjustedLHS <=> AdjustedRHS;
    }
    bool operator==(const Score &Another) const = default;
  };

public:
  struct CommentAssignment {
    /// Since we emit comments as `@comment(i64 INDEX, ...)` (using index
    /// instead of the comment text allows us to avoid invalidating any IR on
    /// a comment edit since it's only read by the backend) we have to report
    /// the index to the callee.
    ///
    /// Also note that it's trivial to go `index -> reference`
    /// (`MyFunction.Comments().at(MyIndex)`), but `reference -> index` is not
    /// possible (`Comments` is a normal vector and we place no limitation
    /// on duplicate elements).
    uint64_t CommentIndex = 0;
    bool LocationMatchesExactly = false;
    Score Score = {};
    const model::StatementComment::TypeOfLocation *ExpectedLocation = nullptr;
  };
  using CommentList = std::vector<CommentAssignment>;

private:
  // This assumes that `Trait::StatementType` is cheap (and, more importantly,
  // safe) to copy
  std::unordered_map<StatementType, CommentList> ResultMap;
  CommentList HomelessComments;

public:
  CommentPlacementHelper() = default;
  CommentPlacementHelper(const model::Function &Function, auto &&Graph) {
    if (Function.Comments().empty()) {
      // No comments in this function, nothing to map
      return;
    }

    // Compute a dominator tree for the given graph to reduce duplicate
    // comments: if a suitable statement is already dominated by another one,
    // we can skip that emission.
    using DominatorTreeType = llvm::DominatorTreeBase<InternalNodeType, false>;
    DominatorTreeType DominatorTree;
    DominatorTree.recalculate(Graph);

    std::vector<std::pair<Score, StatementLocationType>> Scores;
    Scores.resize(Function.Comments().size());

    // First build a score map - select the best location for each comment
    for (auto Node : llvm::depth_first(DominatorTree.getRootNode())) {
      for (const auto &Statement : Trait::getStatements(Node->getBlock())) {
        StatementLocationType Location = Trait::getAddresses(Statement);

        for (auto &&[Index, Comment] : llvm::enumerate(Function.Comments())) {

          // Use the Tversky Index as the scoring function.
          // Note the parameters controlling the weights of the sets: the higher
          // a value is, the bigger the relative negative impact of
          // the corresponding set having *extra* elements compared to the other
          // one.
          //
          // We want this because a line having unaccounted for addresses is a
          // way bigger red flag that a (potentially user edited) comment having
          // them.
          auto Score = tverskyIndex<1, 2>(Comment.Location(), Location);
          if (Score.has_value())
            if (shouldReplace(Scores[Index], { *Score, Location }))
              Scores[Index] = { *Score, Location };
        }
      }
    }

    // Process homeless comments so they don't interfere with the others
    for (auto &&[I, Comment] : llvm::enumerate(Function.Comments()))
      if (Scores[I].first.Numerator == 0)
        HomelessComments.emplace_back(I, false, Score{}, &Comment.Location());

    TreeVisitor Visitor{ ResultMap, Function, Scores };
    Visitor(DominatorTree.getRootNode());
  }

private:
  // Use a dominator tree to build a node-to-comment-list map.
  using DTNode = const llvm::DomTreeNodeBase<InternalNodeType> *;
  struct TreeVisitor {
    std::unordered_map<StatementType, CommentList> &ResultMap;
    const model::Function &Function;
    const std::vector<std::pair<Score, StatementLocationType>> &Scores;

    void operator()(const DTNode &RootNode) {
      impl(RootNode, llvm::SmallBitVector(Scores.size()));
    }

  private:
    RecursiveCoroutine<> impl(const DTNode &Node,
                              llvm::SmallBitVector AssignedInThisBranch) {
      for (const auto &Statement : Trait::getStatements(Node->getBlock())) {
        StatementLocationType Location = Trait::getAddresses(Statement);
        for (auto &&[Index, Comment] : llvm::enumerate(Function.Comments())) {
          if (AssignedInThisBranch.test(Index))
            continue;

          if (Location.empty()) {
            // Skip nodes without locations - they should never have comments
            // assigned to them.
            continue;
          }

          if (Location == Scores[Index].second) {
            // A node with a non-empty location matched the selected score,
            // mark the node as a target for the emission.
            //
            // Note that comments with empty `Scores[Index].second` are
            // homeless.
            bool IsLocationExact = std::ranges::equal(Comment.Location(),
                                                      Scores[Index].second);
            ResultMap[Statement].emplace_back(Index,
                                              IsLocationExact,
                                              Scores[Index].first,
                                              &Comment.Location());

            // Mark this comment as 'skipped' for all the dominated statements.
            AssignedInThisBranch.set(Index);
          }
        }
      }

      // Proceed on the children with a copy of the `AssignedInThisBranch` map
      // so that adjacent children don't affect each other.
      for (const auto &Ch : llvm::children<DTNode>(Node))
        rc_recur impl(Ch, llvm::SmallBitVector{ AssignedInThisBranch });
    }
  };

private:
  bool shouldReplace(std::pair<Score, StatementLocationType> From,
                     std::pair<Score, StatementLocationType> To) {
    const auto &[FromScore, FromLocation] = From;
    const auto &[ToScore, ToLocation] = To;

    // Never replace anything with an empty score
    if (ToScore.Numerator == 0 || ToLocation.empty())
      return false;

    // Always replace an empty score
    if (FromScore.Numerator == 0 || FromLocation.empty())
      return true;

    // If scores differ, that's all we need
    if (ToScore != FromScore)
      return ToScore > FromScore;

    // If scores are indistinguishable, compare locations lexicographically.
    //
    // Note that this will return `false` if both locations are identical, but
    // we're fine with that as in that case both of them will get the comment
    // unless one of them dominates the other.
    auto Comparator = [](const MetaAddress &Left, const MetaAddress &Right) {
      return Left.address() < Right.address();
    };
    return std::ranges::lexicographical_compare(ToLocation,
                                                FromLocation,
                                                Comparator);
  }

public:
  const CommentList &getComments(Trait::StatementType Node) const {
    if (auto Iterator = ResultMap.find(Node); Iterator != ResultMap.end())
      return Iterator->second;

    static CommentList Empty{};
    return Empty;
  }

  const CommentList &getHomelessComments() const { return HomelessComments; }

private:
  static uint64_t relativeComplementSize(std::ranges::range auto &&LHS,
                                         std::ranges::range auto &&RHS) {
    return std::ranges::count_if(LHS, [&RHS](auto &&Element) {
      return !llvm::is_contained(RHS, Element);
    });
  }
  static uint64_t intersectionSize(std::ranges::range auto &&LHS,
                                   std::ranges::range auto &&RHS) {
    return std::ranges::count_if(LHS, [&RHS](auto &&Element) {
      return llvm::is_contained(RHS, Element);
    });
  }

  /// Tversky Index is a more general version of the Jaccard Index commonly
  /// used for evaluation of similarity of set pairs.
  ///
  /// \tparam Alpha weight of the first set
  /// \tparam Beta weight of the second set
  ///
  /// \note If `Alpha == Beta == 1` this produces Jaccard Index.
  ///
  /// \note If `Alpha != Beta`, the index is not symmetric, as such, take care
  ///       to pass LHS and RHS in the exact same order each time.
  ///
  /// \param LHS first set, does *not* have to be sorted.
  /// \param RHS second set, does *not* have to be sorted.
  template<uint64_t Alpha = 1, uint64_t Beta = 1>
  static std::optional<Score>
  tverskyIndex(std::ranges::range auto &&LHS, std::ranges::range auto &&RHS) {
    uint64_t IntersectionSize = intersectionSize(LHS, RHS);
    if (IntersectionSize == 0) {
      // No point proceeding, there's no match.
      return std::nullopt;
    }

    uint64_t LHSComplementSize = relativeComplementSize(LHS, RHS);
    uint64_t RHSComplementSize = relativeComplementSize(RHS, LHS);

    return Score{ .Numerator = IntersectionSize,
                  .Denominator = IntersectionSize + Alpha * LHSComplementSize
                                 + Beta * RHSComplementSize };
  }
};

} // namespace yield
