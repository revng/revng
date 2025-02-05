//
// Copyright rev.ng Labs Srl. See LICENSE.md for details.
//

#include "llvm/ADT/GraphTraits.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/IR/IRBuilder.h"

#include "revng/RestructureCFG/DAGifyPass.h"
#include "revng/RestructureCFG/GenericRegionInfo.h"
#include "revng/RestructureCFG/ScopeGraphGraphTraits.h"
#include "revng/RestructureCFG/ScopeGraphUtils.h"
#include "revng/Support/Assert.h"
#include "revng/Support/Debug.h"
#include "revng/Support/GraphAlgorithms.h"
#include "revng/Support/IRHelpers.h"

using namespace llvm;

// Debug logger
Logger<> DAGifyPassLogger("dagify");

/// Helper function which transform a retreating edge into a `goto` edge
static void processRetreating(const revng::detail::EdgeDescriptor<BasicBlock *>
                                &RetreatingEdge) {
  BasicBlock *Source = RetreatingEdge.first;
  BasicBlock *Target = RetreatingEdge.second;
  revng_log(DAGifyPassLogger,
            "Found retreating " << Source->getName().str() << " -> "
                                << Target->getName().str() << "\n");

  Function *F = Source->getParent();

  // We should not insert a `goto` in a block already containing a
  // `goto_block` marker
  revng_assert(not isGotoBlock(Source));

  // It must be that `Target` is a successor in the `Terminator` of `Source,`
  // and should not be the additional, optional and unique successor
  // representing the `scope_closer` edge, since we are not able to handle it
  // with the following code. In principle we may support this situation, but we
  // have now an invariant of the `ScopeGraph` which states that
  // `scope_closer` edges should not happen to be retreating edges. This
  // is due to the guarantee we have in how we insert such edges in all
  // the passes of our pipeline.
  revng_assert(llvm::any_of(graph_successors(Source),
                            [&Target](const auto &Elem) {
                              return Elem == Target;
                            }));

  // We create a new block which contains only the `goto`, this is an
  // invariant needed in the `ScopeGraph`
  LLVMContext &Context = getContext(F);
  BasicBlock *GotoBlock = BasicBlock::Create(Context,
                                             "goto_" + Target->getName().str(),
                                             F);

  // Connect the `goto` block with the original target
  IRBuilder<> Builder(Context);
  Builder.SetInsertPoint(GotoBlock);
  Builder.CreateBr(Target);

  // Redirect the `Source`->`Target` (may be multiple) edge/s to
  // `Source`->`GotoBlock`. This means that if multiple edges, which are
  // retreating, exist, we will connect them to a single `goto` block,
  // representing the removed retreating. This means that we will need explicit
  // support in the `clifter` pass in order to match this specific topology.
  Instruction *SourceTerminator = Source->getTerminator();
  SourceTerminator->replaceSuccessorWith(Target, GotoBlock);

  // Insert the `goto_block` marker in the `ScopeGraph`
  ScopeGraphBuilder SGBuilder(F);
  SGBuilder.makeGoto(GotoBlock);
}

char DAGifyPass::ID = 0;

static constexpr const char *Flag = "dagify";
using Reg = llvm::RegisterPass<DAGifyPass>;
static Reg X(Flag, "Perform the DAGify pass on the ScopeGrapgh");

class DAGifyPassImpl {
  Function &F;

public:
  DAGifyPassImpl(Function &F) : F(F) {}

public:
  bool run() {

    // Build the `ScopeGraph` on which the `GenericRegionInfo` analysis should
    // be run
    Scope<Function *> ScopeGraph(&F);

    // Build and run the `GenericRegionInfo` analysis on the `ScopeGraph`
    GenericRegionInfo<Scope<Function *>> RegionInfo;
    RegionInfo.clear();
    RegionInfo.compute(ScopeGraph);

    // We keep a boolean variable to track whether the `Module` was modified
    bool ModuleModified = false;

    // We need to perform the DAGify process for each `GenericRegion` that we
    // have identified. We start from the top level `GenericRegion`s, and then
    // we process all the regions nested in a top level one, in a bottom up
    // fashion. Having `GraphTraits` specialized for the tree of
    // `GenericRegion`s, we can handily do this by performing a `post_order`
    // visit on the tree.

    // We keep a global index of the processed region to easy debug
    size_t RegionIndex = 0;
    for (auto &TopLevelRegion : RegionInfo.top_level_regions()) {
      for (auto *Region : post_order(&TopLevelRegion)) {
        revng_log(DAGifyPassLogger,
                  "DAGify processing region with index: "
                    << std::to_string(RegionIndex) << "\n");

        revng_log(DAGifyPassLogger,
                  "The elected head for this region is block"
                    << Region->getHead()->getName().str() << "\n");

        // Each time we attempt to process a `GenericRegion`, we need to
        // recompute the set of retreating edges, since some of the retreating
        // of a `GenericRegion`, may have been already transformed into `goto`
        // edges during the processing of nested `GenericRegion`s.
        // We collect the retreating edges, performing an exploration that
        // starts from the elected `Head` of each identified `GenericRegion`.
        llvm::SmallPtrSet<BasicBlock *, 4> RegionNodes;
        for (auto &RegionNode : Region->blocks()) {
          RegionNodes.insert(RegionNode);
        }

        BasicBlock *Head = Region->getHead();
        using ScopeGT = GraphTraits<Scope<BasicBlock *>>;
        auto Retreatings = getBackedgesWhiteList<BasicBlock *,
                                                 ScopeGT>(Head, RegionNodes);

        // Insert a `goto` in place of each retreating edge
        for (auto &Retreating : Retreatings) {

          // As soon as we find a retreating edge, we mark the `Module` as
          // modified
          ModuleModified = true;

          // Process each retreating edge
          processRetreating(Retreating);
        }

        RegionIndex++;
      }
    }

    // Verify that the output `ScopeGraph` is acyclic, after `DAGify` has
    // processed the input, but only when the `VerifyLog` is enabled
    if (VerifyLog.isEnabled()) {
      revng_assert(isDAG(ScopeGraph));
    }

    return ModuleModified;
  }
};

bool DAGifyPass::runOnFunction(llvm::Function &F) {
  // Log the function name
  revng_log(DAGifyPassLogger,
            "Running DAGify on function " << F.getName().str() << "\n");

  // Instantiate and call the `Impl` class
  DAGifyPassImpl DAGifyImpl(F);
  bool FunctionChanged = DAGifyImpl.run();

  // This pass may transform the CFG by transforming some edges into `goto`
  // edges on the `ScopeGraph`. We propagate the information computed by the
  // `Impl` class.
  return FunctionChanged;
}

void DAGifyPass::getAnalysisUsage(llvm::AnalysisUsage &AU) const {
  // This pass does not preserve the CFG
}
