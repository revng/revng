//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/CFG.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"

#include "revng/RestructureCFG/RemoveIDSArtifactsPass.h"
#include "revng/RestructureCFG/ScopeGraphAlgorithms.h"
#include "revng/RestructureCFG/ScopeGraphGraphTraits.h"
#include "revng/RestructureCFG/ScopeGraphUtils.h"
#include "revng/Support/Assert.h"
#include "revng/Support/Debug.h"
#include "revng/Support/IRHelpers.h"

using namespace llvm;

static Logger Log("remove-ids-artifacts");

using ScopeGraphDomTree = llvm::DomTreeOnView<llvm::BasicBlock, Scope>;
using ScopeGraphPostDomTree = llvm::PostDomTreeOnView<llvm::BasicBlock, Scope>;

//===-------------------------- Helper functions --------------------------===//

/// The `SwitchInst` terminating `BB`, if any, `nullptr` otherwise.
static SwitchInst *getSwitch(BasicBlock *BB) {
  return dyn_cast<SwitchInst>(BB->getTerminator());
}

/// The only block that can branch to `BB`. `nullptr` when there none or many.
static BasicBlock *getSingleCFGPredecessor(BasicBlock *BB) {
  BasicBlock *Result = nullptr;

  for (BasicBlock *Predecessor : predecessors(BB)) {
    if (Result == nullptr)
      Result = Predecessor;
    else if (Result != Predecessor)
      return nullptr;
  }

  return Result;
}

/// For every block of a function, the blocks holding a `scope_closer` aimed at
/// it, in block order.
///
/// The matcher asks this of every block of the `ScopeGraph` and each fold asks
/// it again, so answering one question at a time by walking the function is
/// quadratic in the number of blocks. One walk answers all of them.
///
/// Walking the function also settles the order for free. It does not depend on
/// a use list or on a block having a name, so a run is deterministic and
/// reproducible down to its log, and it visits every block once, so a block
/// that is both a plain CFG predecessor of its target and the source of the
/// `scope_closer` is listed once rather than twice and `eraseScopeCloser` is
/// never asked to cut the same marker twice.
class ScopeCloserPredecessors {
private:
  DenseMap<const BasicBlock *, SmallVector<BasicBlock *>> Sources;

public:
  explicit ScopeCloserPredecessors(Function &F) {
    for (BasicBlock &BB : F)
      if (BasicBlock *Target = getScopeCloserTarget(&BB))
        Sources[Target].push_back(&BB);
  }

public:
  /// The blocks holding a `scope_closer` aimed at `BB`, in block order.
  ArrayRef<BasicBlock *> operator[](const BasicBlock *BB) const {
    auto It = Sources.find(BB);
    if (It == Sources.end())
      return {};

    return It->second;
  }
};

/// The block where `Switch` directs control flow to when the condition is
/// equal to `Label`.
static BasicBlock *dispatch(SwitchInst *Switch, ConstantInt *Label) {
  auto Case = Switch->findCaseValue(Label);
  if (Case == Switch->case_default())
    return Switch->getDefaultDest();

  return Case->getCaseSuccessor();
}

/// The one block of `F` with no `ScopeGraph` successors.
///
/// `enforce-single-exit` runs before this pass in the pipeline, so there is
/// exactly one when the pass starts, and the pass keeps it that way.
static BasicBlock *getSingleScopeGraphExit(Function &F) {
  BasicBlock *Result = nullptr;

  for (BasicBlock &BB : F) {
    if (not getScopeGraphSuccessors(&BB).empty())
      continue;

    revng_assert(Result == nullptr,
                 "the ScopeGraph has more than one exit on entry");
    Result = &BB;
  }

  revng_assert(Result != nullptr, "the ScopeGraph has no exit");
  return Result;
}

//===----------------------------- The matcher ----------------------------===//

/// Returns the predecessor block that `Node` can be squashed into, or `nullptr`
/// when it cannot.
///
/// `Node` qualifies when all of the following hold.
///
/// - It branches more than one way on the `ScopeGraph`. A `switch` that IDS
///   left with a single destination has already been turned into a branch, so
///   this only ever rules out a shape this pass has no business touching.
/// - Its terminator is a `switch`.
/// - It holds nothing but that terminator. There is then no instruction in it
///   anything else could depend on, which is what makes it safe to erase, and
///   it also rules out the `scope_closer` and `goto_block` markers, since those
///   are calls sitting before the terminator.
/// - On the CFG, it is reachable from a single predecessor block, so erasing it
///   cannot cut the path of an unrelated predecessor.
/// - That predecessor block ends in a `switch` on the very same SSA value,
///   which is what makes predecessor absorb the successor.
/// - A `scope_closer` aims at it, and every block holding one is dominated by
///   that predecessor on the `ScopeGraph`. The fold retargets all of them at
///   once, so a source reachable without going through the predecessor would
///   have the scope it closes rewritten by a fold that has nothing to do with
///   the path reaching it.
///
/// That last condition is what tells an IDS `Tail` from an IDS `Head`, and
/// nothing else does: `createHead` clones the terminator of the conditional
/// into a fresh block exactly like `performMultipleIDS` does for the `Tail`, so
/// on the CFG the two are the same shape. `performMultipleIDS` closes every
/// divergent exit into the `Tail` it just created, and nothing is ever closed
/// into a `Head`.
///
/// Folding a `Head` away would be correct on the CFG and wrong for the scope
/// structure. A `Head` is there to give a single entry to a divergent scope
/// entered through several successors of the original switch; squashing it
/// would move back those successors back to the conditional; if the successors
/// have a least common postdominator that is strictly post dominated by the
/// immediate postdominator of the conditional that would cause the ScopeGraph
/// not to be fully decided anymore and the Clifter to crash.
static BasicBlock *
getFoldablePredecessor(BasicBlock *Node,
                       const ScopeGraphDomTree &DomTree,
                       const ScopeCloserPredecessors &Closers) {
  SwitchInst *NodeSwitch = getSwitch(Node);
  if (NodeSwitch == nullptr)
    return nullptr;

  if (getScopeGraphSuccessors(Node).size() < 2)
    return nullptr;

  if (Node->size() != 1)
    return nullptr;

  BasicBlock *Predecessor = getSingleCFGPredecessor(Node);
  if (Predecessor == nullptr or Predecessor == Node)
    return nullptr;

  SwitchInst *PredecessorSwitch = getSwitch(Predecessor);
  if (PredecessorSwitch == nullptr)
    return nullptr;

  if (PredecessorSwitch->getCondition() != NodeSwitch->getCondition())
    return nullptr;

  ArrayRef<BasicBlock *> Sources = Closers[Node];

  if (Sources.empty()) {
    revng_log(Log,
              Node->getName() << " is not closed into, so it is an IDS head "
                              << "rather than a tail");
    return nullptr;
  }

  for (BasicBlock *Source : Sources) {
    if (not DomTree.dominates(Predecessor, Source)) {
      revng_log(Log,
                "the scope_closer in "
                  << Source->getName() << " aimed at " << Node->getName()
                  << " is not dominated by " << Predecessor->getName());
      return nullptr;
    }
  }

  return Predecessor;
}

//===------------------------------ The fold ------------------------------===//

/// One fold, as the sweep over the `ScopeGraph` found it.
struct FoldCandidate {
  /// The block that goes away.
  BasicBlock *Node;

  /// The only block that can branch to `Node`, which absorbs it.
  BasicBlock *Predecessor;

  /// Where the `scope_closer`s aimed at `Node` are pointed instead.
  BasicBlock *Continuation;
};

/// Squash `Node` into a foldable `Predecessor`, and erase it.
/// Redirects to `Continuation` all the scope closer edges aimed at `Node`.
///
/// Both blocks `switch` on the same value, and `Node` is reached from
/// `Predecessor` alone, so on every edge from `Predecessor` to `Node` the value
/// of the condition is already pinned down by the edge itself. Substituting
/// what `Node` does with that value into `Predecessor` is then a local rewrite:
///
/// - an edge labelled `v` reaching `Node` ends up wherever `Node` sends `v`;
/// - the default edge, when it reaches `Node`, carries every value no case of
///   `Predecessor` mentions, so `Node` keeps dispatching those: its default
///   becomes the default, and each of its cases a case, except for the labels
///   `Predecessor` already handles, which never arrive through the default.
///
/// Every case label of `Predecessor` is kept, even one resolving to where the
/// default goes. Dropping those is a separate transformation and it could leave
/// the `switch` with a single successor, which the `ScopeGraph` treats
/// differently.
///
static void fold(const ScopeGraphBuilder &SGBuilder,
                 const ScopeCloserPredecessors &Closers,
                 const FoldCandidate &Candidate) {
  const auto &[Node, Predecessor, Continuation] = Candidate;

  revng_log(Log,
            "Folding " << Node->getName() << " into "
                       << Predecessor->getName());
  LoggerIndent Indent{ Log };

  SwitchInst *NodeSwitch = getSwitch(Node);
  SwitchInst *PredecessorSwitch = getSwitch(Predecessor);
  bool NodeIsDefault = PredecessorSwitch->getDefaultDest() == Node;

  for (BasicBlock *Source : Closers[Node]) {
    revng_log(Log,
              "retargeting the scope_closer in " << Source->getName() << " to "
                                                 << Continuation->getName());

    // A block carrying a `scope_closer` has a `ScopeGraph` successor, so it is
    // never the exit, and this cannot build a self loop
    revng_assert(Source != Continuation);

    SGBuilder.eraseScopeCloser(Source);
    SGBuilder.addScopeCloser(Source, Continuation);
  }

  // The cases of `Predecessor` in the order they already have, then the ones
  // `Node` contributes in theirs, so the result does not depend on anything but
  // the two `switch`es
  SmallVector<std::pair<ConstantInt *, BasicBlock *>> Cases;
  SmallPtrSet<ConstantInt *, 8> Handled;

  for (const auto &Case : PredecessorSwitch->cases()) {
    ConstantInt *Label = Case.getCaseValue();
    BasicBlock *Destination = Case.getCaseSuccessor();

    // `ConstantInt`s are uniqued per type and value, so equal labels are the
    // same pointer
    Handled.insert(Label);

    if (Destination == Node)
      Destination = dispatch(NodeSwitch, Label);

    Cases.emplace_back(Label, Destination);
  }

  if (NodeIsDefault) {
    for (const auto &Case : NodeSwitch->cases()) {
      if (not Handled.contains(Case.getCaseValue()))
        Cases.emplace_back(Case.getCaseValue(), Case.getCaseSuccessor());
    }
  }

  BasicBlock *Default = NodeIsDefault ? NodeSwitch->getDefaultDest() :
                                        PredecessorSwitch->getDefaultDest();

  revng_log(Log,
            "the folded switch has " << Cases.size()
                                     << " cases and defaults to "
                                     << Default->getName());

  // Built before anything is erased, so that the destinations stay valid while
  // it is assembled
  auto *Folded = SwitchInst::Create(PredecessorSwitch->getCondition(),
                                    Default,
                                    Cases.size());
  Folded->setDebugLoc(PredecessorSwitch->getDebugLoc());
  for (const auto &[Label, Destination] : Cases)
    Folded->addCase(Label, Destination);

  ReplaceInstWithInst(PredecessorSwitch, Folded);

  // The `scope_closer`s are gone, but the `BlockAddress` they used may still be
  // around with no users left. Drop it, so that the block is erased without an
  // address taken.
  if (BlockAddress *Address = BlockAddress::lookup(Node))
    if (Address->use_empty())
      Address->destroyConstant();

  // Nothing targets `Node` any more: the folded `switch` resolved every edge
  // that used to reach it, and it had no other predecessor
  DeleteDeadBlocks(SmallVector<BasicBlock *>{ Node });
}

//===------------------------------ The pass ------------------------------===//

bool removeIDSArtifacts(Function &F) {
  revng_log(Log, "Running RemoveIDSArtifacts on " << F.getName());
  LoggerIndent Indent{ Log };

  if (VerifyLog.isEnabled()) {
    for (BasicBlock &BB : F)
      revng_assert(BB.phis().empty(), "RemoveIDSArtifacts expects no PHINodes");
  }

  Scope<Function *> ScopeGraph(&F);

  // The matcher asks which blocks a candidate's predecessor dominates and
  // which blocks close into the candidate, and the sweep below changes neither,
  // so one tree and one map answer for every candidate.
  ScopeGraphDomTree DomTree;
  DomTree.recalculate(F);

  const ScopeCloserPredecessors Closers(F);

  // Compute Candidates in post order, so that a block is guaranteed to be
  // folded into its predecessor before the predecessor itself is considered for
  // further folding.
  // All the candidates are computed in one sweep, before a single block is
  // folded.
  SmallVector<FoldCandidate> Candidates;
  DenseMap<const BasicBlock *, unsigned> Position;
  for (BasicBlock *Node : post_order(ScopeGraph)) {
    if (BasicBlock *Predecessor = getFoldablePredecessor(Node,
                                                         DomTree,
                                                         Closers)) {
      Position[Node] = Candidates.size();
      Candidates.push_back({ Node, Predecessor, nullptr });
    }
  }

  if (Candidates.empty()) {
    revng_log(Log, "nothing to fold");
    return false;
  }

  /// Whether `BB` is folded away before the fold at `Index`, which is the one
  /// thing that would leave this holding a pointer to an erased block.
  auto IsGoneBy = [&Position](const BasicBlock *BB, unsigned Index) {
    auto It = Position.find(BB);
    return It != Position.end() and It->second < Index;
  };

  // We can compute the postdominator tree once. All the information that is
  // necessary for folding can be computed beforehand using this, without other
  // updates.
  ScopeGraphPostDomTree PostDomTree;
  PostDomTree.recalculate(F);

  // The ScopeGraph is guaranteed to have a single exit at the beginning, and we
  // we need to preserve it.
  BasicBlock *SingleExit = getSingleScopeGraphExit(F);

  for (auto &&[Index, Candidate] : llvm::enumerate(Candidates)) {
    // Candidate.Node is a switch here, and is scheduled for folding into its
    // CFG predecessor. In order to do so we need to compute where the scopes
    // that were opening in Candidate.Node would close.
    // The scope closer edges that aim at Candidate.Node before folding are
    // guaranteed to be dominated by Candidate.Predecessor, and represent the
    // end of scopes that open in Candidate.Predecessor.
    // Folding Candidate.Node into Candidate.Predecessor needs to rearrange the
    // scopes that open in Candidate.Node and Candidate.Predecessor as if they
    // all open in Candidate.Predecessor and close in the immediate
    // postdominator of Candidate.Node, if any.
    BasicBlock *Continuation = SingleExit;
    if (auto *Node = PostDomTree[Candidate.Node])
      if (auto *IDom = Node->getIDom())
        if (IDom->getBlock() != nullptr)
          Continuation = IDom->getBlock();

    // Lookup the Continuation that will survive all the folds.
    while (IsGoneBy(Continuation, Index))
      Continuation = Candidates[Position[Continuation]].Continuation;

    Candidate.Continuation = Continuation;

    revng_assert(not IsGoneBy(Candidate.Predecessor, Index),
                 "a block is folded into a block already folded away");

    // Each fold reads the blocks closing into its own node out of a map built
    // before any of them ran, and the only thing a fold does to that map is
    // move sources onto its `Continuation`. The chase above lands on a block
    // that no fold removes, so no fold changes what a later one reads.
    revng_assert(Position.count(Continuation) == 0,
                 "a scope_closer is retargeted at a block folded later");
  }

  const ScopeGraphBuilder SGBuilder(&F);
  for (const FoldCandidate &Candidate : Candidates)
    fold(SGBuilder, Closers, Candidate);

  // The whole point of retargeting the `scope_closer`s rather than cutting them
  // is that the exit stays the only one, so it is worth checking when
  // verification is on
  if (VerifyLog.isEnabled())
    revng_assert(getSingleScopeGraphExit(F) == SingleExit);

  return true;
}

char RemoveIDSArtifactsLegacyPass::ID = 0;
static constexpr const char *Flag = "remove-ids-artifacts";
using Register = RegisterPass<RemoveIDSArtifactsLegacyPass>;
static Register
  X(Flag, "Collapse the switch chains left behind by inline-divergent-scopes");
