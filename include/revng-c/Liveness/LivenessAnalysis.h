//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#ifndef REVNGC_LIVENESS_ANALYSIS_H
#define REVNGC_LIVENESS_ANALYSIS_H

// LLVM includes
#include <llvm/IR/Function.h>

// revng includes
#include <revng/Support/IRHelpers.h>
#include <revng/Support/CommandLine.h>
#include <revng/Support/MonotoneFramework.h>

namespace LivenessAnalysis {

using LiveSet = UnionMonotoneSet<llvm::Instruction *>;
using LivenessMap =  std::map<llvm::BasicBlock *, LiveSet>;

class Analysis
  : public MonotoneFramework<Analysis,
                             llvm::BasicBlock *,
                             LiveSet,
                             VisitType::PostOrder,
                             llvm::SmallVector<llvm::BasicBlock *, 2>> {
private:
  llvm::Function &F;
  LivenessMap LiveIn;
  using UseSet = std::set<llvm::Use *>;
  std::map<llvm::BasicBlock *, std::map<llvm::BasicBlock *, UseSet>> PHIEdges;

public:
  using Base = MonotoneFramework<Analysis,
                                 llvm::BasicBlock *,
                                 LiveSet,
                                 VisitType::PostOrder,
                                 llvm::SmallVector<llvm::BasicBlock *, 2>>;

  Analysis(llvm::Function &F) :
    Base(&F.getEntryBlock()),
    F(F) {
    for (llvm::BasicBlock &BB : F) {
      auto NextSucc = llvm::succ_begin(&BB);
      auto EndSucc = llvm::succ_end(&BB);
      if (NextSucc == EndSucc) // BB has no successors
        Base::registerExtremal(&BB);
    }
  }

  void initialize() {
    Base::initialize();
    LiveIn.clear();
    PHIEdges.clear();
  }

  void assertLowerThanOrEqual(const LiveSet &A,
                              const LiveSet &B) const {
    revng_assert(A.lowerThanOrEqual(B));
  }

  /// This Analysis uses DefaultInterrupt, hence it is never supposed to dump
  /// the final state.
  void dumpFinalState() const { revng_abort(); }

  /// Gets the predecessor BasicBlock in the CFG. Being a backward analysis the
  /// 'successors' in analysis order are the 'predecessor' in CFG order.
  llvm::SmallVector<llvm::BasicBlock *, 2>
  successors(llvm::BasicBlock *BB, InterruptType &) const {
    llvm::SmallVector<llvm::BasicBlock *, 2> Result;
    for (llvm::BasicBlock *Pred : make_range(pred_begin(BB), pred_end(BB)))
      Result.push_back(Pred);
    return Result;
  }

  size_t successor_size(llvm::BasicBlock *BB, InterruptType &) const {
    return succ_end(BB) - succ_begin(BB);
  }

  static LiveSet extremalValue(llvm::BasicBlock *BB) {
    return LiveSet::bottom();
  }

  /// \brief Gets the final results of the analysis
  ///
  /// returns a LivenessMap, mapping each BasicBlock to its LiveIn set,
  /// representing all the Instructions that are live at the beginning.
  ///
  /// NOTE: we only track Instructions, because anything that is not an
  /// Instruction is always live.
  const LivenessMap &getLiveIn() const {
    return LiveIn;
  };

  /// \brief Extracts the final results of the analysis.
  ///
  /// Returns a LivenessMap, mapping each BasicBlock to its LiveIn set,
  /// representing all the Instructions that are live at the beginning.
  ///
  /// NOTE: The LiveIn is moved from, so it's left is undetermined state
  ///       after a call to this method.
  LivenessMap &&extractLiveIn() {
    return std::move(LiveIn);
  };

  // ---- Transfer function and handleEdge, to propagate the analysis ----

  InterruptType transfer(llvm::BasicBlock *BB);

  llvm::Optional<LiveSet>
  handleEdge(const LiveSet &Original,
             llvm::BasicBlock *Source,
             llvm::BasicBlock *Destination) const;
};

} // end namespace LivenessAnalysis

#endif // REVNGC_LIVENESS_ANALYSIS_H
